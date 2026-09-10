/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.compute.lucene.query;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.compute.lucene.ShardContext;
import org.elasticsearch.core.IOUtils;
import org.elasticsearch.search.DocValueFormat;
import org.elasticsearch.search.sort.FieldSortBuilder;
import org.elasticsearch.search.sort.ScoreSortBuilder;
import org.elasticsearch.search.sort.SortAndFormats;
import org.elasticsearch.search.sort.SortBuilder;
import org.elasticsearch.test.ESTestCase;
import org.junit.After;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

import static org.elasticsearch.compute.lucene.query.LuceneSliceQueue.PartitioningStrategy.DOC;
import static org.elasticsearch.compute.lucene.query.LuceneSliceQueue.PartitioningStrategy.SEGMENT;
import static org.elasticsearch.compute.lucene.query.LuceneSliceQueue.PartitioningStrategy.SHARD;
import static org.hamcrest.Matchers.equalTo;

/**
 * Tests {@link LuceneTopNSourceOperator#autoStrategy} / {@link LuceneTopNSourceOperator#pickStrategy}: a field-sorted
 * TopN goes to {@link LuceneSliceQueue.PartitioningStrategy#DOC} unless the sort can be short-circuited (a
 * points-indexed sort field, a sort congruent with the index sort, or a low-cost query), in which case
 * {@link LuceneSliceQueue.PartitioningStrategy#SEGMENT} is kept.
 *
 * <p>A {@code _score} sort instead picks between {@link LuceneSliceQueue.PartitioningStrategy#SHARD} and
 * {@link LuceneSliceQueue.PartitioningStrategy#SEGMENT} by query cost — see {@link LuceneTopNSourceOperator#scoringStrategy}.
 *
 * <p>Docs carry {@code kw} (keyword: postings + {@link SortedDocValuesField}, no points, unique per doc), {@code num}
 * (long: points + doc values), {@code common} (keyword with the same value in every doc) and {@code t}/{@code t2}
 * (analyzed text, so they carry positions). {@code kw} exercises the scan-dominant path and the cheap end of the cost
 * gate, {@code num} the points path, {@code common} the expensive end of the cost gate, and {@code t}/{@code t2} the
 * positional queries whose cost understates their work.
 */
public class LuceneTopNSourceOperatorAutoStrategyTests extends ESTestCase {

    private static final int NUM_DOCS = 200;

    private Directory directory;
    private IndexReader reader;

    @After
    public void closeIndex() throws Exception {
        IOUtils.close(reader, directory);
    }

    private static String kw(int i) {
        return String.format(Locale.ROOT, "%04d", i);
    }

    /** Builds an index (optionally index-sorted) and a {@link ShardContext} whose {@code buildSort} returns {@code searchSort}. */
    private ShardContext context(Sort indexSort, SortAndFormats searchSort) throws IOException {
        directory = newDirectory();
        IndexWriterConfig iwc = new IndexWriterConfig();
        if (indexSort != null) {
            iwc.setIndexSort(indexSort);
        }
        try (IndexWriter writer = new IndexWriter(directory, iwc)) {
            for (int i = 0; i < NUM_DOCS; i++) {
                Document doc = new Document();
                doc.add(new StringField("kw", kw(i), Field.Store.NO));
                doc.add(new SortedDocValuesField("kw", new BytesRef(kw(i))));
                doc.add(new LongPoint("num", i));
                doc.add(new NumericDocValuesField("num", i));
                doc.add(new StringField("common", "y", Field.Store.NO));
                // Analyzed, so they carry positions: "common" is in every doc, "rare" in a fifth of them, so
                // the phrase "rare common" costs only what "rare" costs.
                String text = i % 5 == 0 ? "rare common" : "filler common";
                doc.add(new TextField("t", text, Field.Store.NO));
                doc.add(new TextField("t2", text, Field.Store.NO));
                writer.addDocument(doc);
            }
        }
        reader = DirectoryReader.open(directory);
        return new LuceneSourceOperatorTests.MockShardContext(reader, 0) {
            @Override
            public Optional<SortAndFormats> buildSort(List<SortBuilder<?>> sorts) {
                return Optional.ofNullable(searchSort);
            }
        };
    }

    private static SortAndFormats sortAndFormats(Sort sort) {
        return new SortAndFormats(sort, new DocValueFormat[] { DocValueFormat.RAW });
    }

    private static List<SortBuilder<?>> fieldSort(String field) {
        return List.of(new FieldSortBuilder(field));
    }

    // --- pickStrategy: scan-dominant field sort -> DOC ---

    public void testKeywordSortScanDominantPicksDoc() throws IOException {
        ShardContext ctx = context(null, null);
        // kw has no points, no congruent index sort, and a MatchAll cost (NUM_DOCS) above the threshold.
        assertThat(pick(ctx, Queries.ALL_DOCS_INSTANCE, "kw", 1), equalTo(DOC));
    }

    // --- pickStrategy: points-indexed sort field -> SEGMENT ---

    public void testPointsSortFieldPicksSegment() throws IOException {
        ShardContext ctx = context(null, null);
        // num has a points index -> the sort prunes via the BKD tree, DOC would break it.
        assertThat(pick(ctx, Queries.ALL_DOCS_INSTANCE, "num", 1), equalTo(SEGMENT));
    }

    // --- pickStrategy: sort congruent with the index sort -> SEGMENT ---

    public void testSortCongruentWithIndexSortPicksSegment() throws IOException {
        Sort sort = new Sort(new SortField("kw", SortField.Type.STRING));
        ShardContext ctx = context(sort, sortAndFormats(sort));
        // kw has no points, but the search sort equals the index sort -> Lucene early-terminates.
        assertThat(pick(ctx, Queries.ALL_DOCS_INSTANCE, "kw", 1), equalTo(SEGMENT));
    }

    public void testSortNotCongruentWithIndexSortPicksDoc() throws IOException {
        Sort indexSort = new Sort(new SortField("kw", SortField.Type.STRING));
        // Index sorted by kw, but the query sorts by a different (reverse) order -> not congruent.
        Sort searchSort = new Sort(new SortField("kw", SortField.Type.STRING, true));
        ShardContext ctx = context(indexSort, sortAndFormats(searchSort));
        assertThat(pick(ctx, Queries.ALL_DOCS_INSTANCE, "kw", 1), equalTo(DOC));
    }

    // --- pickStrategy: low-cost query -> SEGMENT ---

    public void testLowCostQueryPicksSegment() throws IOException {
        ShardContext ctx = context(null, null);
        Query oneDoc = new TermQuery(new Term("kw", kw(7)));
        // The filter matches a single doc, well below minCostForDoc -> not worth DOC's overhead.
        assertThat(pick(ctx, oneDoc, "kw", 1_000_000), equalTo(SEGMENT));
    }

    public void testHighCostQueryPicksDoc() throws IOException {
        ShardContext ctx = context(null, null);
        // MatchAll matches every doc; above the (tiny) threshold -> DOC.
        assertThat(pick(ctx, Queries.ALL_DOCS_INSTANCE, "kw", NUM_DOCS / 2), equalTo(DOC));
    }

    // --- pickStrategy: costly-to-build filter clause -> SEGMENT (even with a scan-dominant field sort) ---

    public void testCostlyFilterClausePicksSegment() throws IOException {
        ShardContext ctx = context(null, null);
        // Sort by kw (scan-dominant, no points), but the WHERE is a point range: costly to build a scorer for, so DOC's
        // sub-segment slices would each pay the full-segment BKD cost -> keep SEGMENT.
        assertThat(pick(ctx, LongPoint.newRangeQuery("num", 5, 10), "kw", 1), equalTo(SEGMENT));
    }

    // --- autoStrategy wrapper: non-field sorts -> SEGMENT ---

    public void testNonFieldPrimarySortPicksSegment() throws IOException {
        ShardContext ctx = context(null, null);
        assertThat(auto(ctx, List.of(new ScoreSortBuilder()), false, 1, Queries.ALL_DOCS_INSTANCE), equalTo(SEGMENT));
        assertThat(auto(ctx, List.of(), false, 1, Queries.ALL_DOCS_INSTANCE), equalTo(SEGMENT));
    }

    public void testFieldSortThroughAutoStrategyPicksDoc() throws IOException {
        ShardContext ctx = context(null, null);
        // End-to-end through the public entry point: keyword sort, scan-dominant -> DOC.
        assertThat(auto(ctx, fieldSort("kw"), false, 1, Queries.ALL_DOCS_INSTANCE), equalTo(DOC));
    }

    // --- scoringStrategy: _score sorts are cost-gated between SHARD and SEGMENT, never DOC ---

    public void testScoreSortMatchAllPicksShard() throws IOException {
        ShardContext ctx = context(null, null);
        // Every score is 1.0, so the TopN is arbitrary and fanning out buys nothing. Note the tiny threshold: this is
        // decided before the cost gate.
        assertThat(scoring(ctx, Queries.ALL_DOCS_INSTANCE, 1), equalTo(SHARD));
    }

    public void testScoreSortMatchNonePicksShard() throws IOException {
        ShardContext ctx = context(null, null);
        assertThat(scoring(ctx, Queries.NO_DOCS_INSTANCE, 1), equalTo(SHARD));
    }

    public void testScoreSortCostlyClausePicksSegment() throws IOException {
        ShardContext ctx = context(null, null);
        // A point range builds its scorer per segment either way, so SHARD would only serialise it. The huge threshold
        // shows the costly-clause check wins over the cost gate.
        assertThat(scoring(ctx, LongPoint.newRangeQuery("num", 5, 10), 1_000_000), equalTo(SEGMENT));
    }

    public void testScoreSortCheapQueryPicksShard() throws IOException {
        ShardContext ctx = context(null, null);
        // One matching doc out of NUM_DOCS: the SEGMENT fan-out costs more than the scan it parallelizes.
        assertThat(scoring(ctx, new TermQuery(new Term("kw", kw(7))), NUM_DOCS / 2), equalTo(SHARD));
    }

    public void testScoreSortExpensiveQueryPicksSegment() throws IOException {
        ShardContext ctx = context(null, null);
        // "common" matches every doc, so the cost clears the threshold and the scoring scan is worth parallelizing.
        assertThat(scoring(ctx, new TermQuery(new Term("common", "y")), NUM_DOCS / 2), equalTo(SEGMENT));
    }

    public void testScoreSortPhraseQueryPicksSegment() throws IOException {
        ShardContext ctx = context(null, null);
        // A phrase is a conjunction, so its cost is that of its rarest term — here "rare", in a fifth of the docs,
        // which is below the threshold and would read as cheap. But verifying the phrase decodes and intersects
        // positions for each of those candidates, so this is not a query to run on a single driver.
        Query phrase = new PhraseQuery("t", "rare", "common");
        assertThat(scoring(ctx, phrase, NUM_DOCS), equalTo(SEGMENT));
        // The equivalent disjunction sums its terms' costs instead, so the gate already sends it to SEGMENT. The
        // point of the assertion above is that the phrase agrees, rather than reading as cheap and going to SHARD.
        Query disjunction = new BooleanQuery.Builder().add(new TermQuery(new Term("t", "rare")), BooleanClause.Occur.SHOULD)
            .add(new TermQuery(new Term("t", "common")), BooleanClause.Occur.SHOULD)
            .build();
        assertThat(scoring(ctx, disjunction, NUM_DOCS), equalTo(SEGMENT));
    }

    public void testScoreSortNestedPhraseQueryPicksSegment() throws IOException {
        ShardContext ctx = context(null, null);
        // `MATCH_PHRASE(a, "...") OR MATCH_PHRASE(b, "...")` — the shape the full-text benchmark runs. The phrases
        // are nested inside a BooleanQuery, so the check has to walk the tree rather than test the root.
        Query nested = new BooleanQuery.Builder().add(new PhraseQuery("t", "rare", "common"), BooleanClause.Occur.SHOULD)
            .add(new PhraseQuery("t2", "rare", "common"), BooleanClause.Occur.SHOULD)
            .build();
        assertThat(scoring(ctx, nested, NUM_DOCS), equalTo(SEGMENT));
    }

    public void testScoreSortSingleTermPhraseQueryPicksShard() throws IOException {
        ShardContext ctx = context(null, null);
        // Lucene rewrites a one-term phrase to a TermQuery, which has no positions to verify. It really is cheap,
        // so it should still reach SHARD — the positional check must not swallow the whole optimisation.
        Query onePosition = new PhraseQuery("t", "rare");
        assertThat(scoring(ctx, onePosition, NUM_DOCS), equalTo(SHARD));
    }

    public void testScoreSortThroughAutoStrategyIgnoresTheSort() throws IOException {
        ShardContext ctx = context(null, null);
        // needsScore short-circuits the sort analysis: a points-indexed primary sort field would otherwise force
        // SEGMENT, but a cheap scored query still goes to SHARD.
        Query oneDoc = new TermQuery(new Term("kw", kw(7)));
        assertThat(auto(ctx, fieldSort("num"), true, NUM_DOCS / 2, oneDoc), equalTo(SHARD));
        assertThat(auto(ctx, List.of(new ScoreSortBuilder()), true, NUM_DOCS / 2, oneDoc), equalTo(SHARD));
    }

    private static LuceneSliceQueue.PartitioningStrategy pick(ShardContext ctx, Query query, String field, long minCostForDoc) {
        return LuceneTopNSourceOperator.pickStrategy(ctx, query, field, fieldSort(field), minCostForDoc);
    }

    /** Rewrites first, as {@link LuceneSliceQueue} does before it picks, so {@code queryCost}'s assertion holds. */
    private static LuceneSliceQueue.PartitioningStrategy scoring(ShardContext ctx, Query query, long minCostForDoc) throws IOException {
        return LuceneTopNSourceOperator.scoringStrategy(ctx, ctx.searcher().rewrite(query), minCostForDoc);
    }

    private static LuceneSliceQueue.PartitioningStrategy auto(
        ShardContext ctx,
        List<SortBuilder<?>> sorts,
        boolean needsScore,
        long minCostForDoc,
        Query query
    ) {
        return LuceneTopNSourceOperator.autoStrategy(sorts, needsScore, minCostForDoc)
            .pickStrategy(LuceneOperator.NO_LIMIT)
            .apply(ctx, query);
    }
}
