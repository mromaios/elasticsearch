/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

package org.elasticsearch.xpack.core.inference.chunking;

import com.ibm.icu.text.BreakIterator;

import org.elasticsearch.common.Strings;
import org.elasticsearch.inference.ChunkingSettings;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Split text into chunks recursively based on a list of separator regex strings.
 * The maximum chunk size is measured in words and controlled
 * by {@code maxNumberWordsPerChunk}. For each separator the chunker will go through the following process:
 * 1. Split the text on each regex match of the separator.
 * 2. For each chunk after the merge:
 *     1. Return it if it is within the maximum chunk size.
 *     2. Repeat the process using the next separator in the list if the chunk exceeds the maximum chunk size.
 *     If there are no more separators left to try, run the {@code SentenceBoundaryChunker} with the provided
 *     max chunk size and no overlaps.
 */
public class RecursiveChunker implements Chunker {
    private final BreakIterator wordIterator;

    public RecursiveChunker() {
        wordIterator = BreakIterator.getWordInstance();
    }

    @Override
    public List<ChunkOffset> chunk(String input, ChunkingSettings chunkingSettings) {
        if (chunkingSettings instanceof RecursiveChunkingSettings recursiveChunkingSettings) {
            return chunk(
                input,
                new ChunkOffset(0, input.length()),
                recursiveChunkingSettings.getSeparators(),
                recursiveChunkingSettings.maxChunkSize()
            );
        } else {
            throw new IllegalArgumentException(
                Strings.format("RecursiveChunker can't use ChunkingSettings with strategy [%s]", chunkingSettings.getChunkingStrategy())
            );
        }
    }

    /**
     * Splits {@code initialOffset} down until every emitted chunk is within {@code maxChunkSize}, advancing to the next
     * separator each time a chunk is still too large.
     * <p>
     * The descent is driven by an explicit stack. Chunks are pushed in reverse so that they pop in document order,
     * preserving the emission order of the recursive chunking strategy.
     */
    private List<ChunkOffset> chunk(String input, ChunkOffset initialOffset, List<String> separators, int maxChunkSize) {
        if (initialOffset.start() == initialOffset.end()) {
            return List.of(initialOffset);
        }

        // The word count for each pending chunk is carried alongside it, because splitting and merging already compute it.
        var pendingChunks = new ArrayDeque<PendingChunk>();
        pendingChunks.push(new PendingChunk(buildChunkOffsetAndCount(input, initialOffset), 0));

        var chunks = new ArrayList<ChunkOffset>();
        while (pendingChunks.isEmpty() == false) {
            var pendingChunk = pendingChunks.pop();
            var chunkOffsetAndCount = pendingChunk.chunkOffsetAndCount();
            var offset = chunkOffsetAndCount.chunkOffset();

            if (offset.start() == offset.end() || isChunkWithinMaxSize(chunkOffsetAndCount, maxChunkSize)) {
                chunks.add(offset);
                continue;
            }

            if (pendingChunk.separatorIndex() > separators.size() - 1) {
                chunks.addAll(chunkWithBackupChunker(input, offset, maxChunkSize));
                continue;
            }

            var potentialChunks = mergeChunkOffsetsUpToMaxChunkSize(
                splitTextBySeparatorRegex(input, offset, separators.get(pendingChunk.separatorIndex())),
                maxChunkSize
            );
            for (int i = potentialChunks.size() - 1; i >= 0; i--) {
                pendingChunks.push(new PendingChunk(potentialChunks.get(i), pendingChunk.separatorIndex() + 1));
            }
        }

        return chunks;
    }

    private boolean isChunkWithinMaxSize(ChunkOffsetAndCount chunkOffsetAndCount, int maxChunkSize) {
        return chunkOffsetAndCount.wordCount <= maxChunkSize;
    }

    private ChunkOffsetAndCount buildChunkOffsetAndCount(String fullText, ChunkOffset offset) {
        wordIterator.setText(fullText);
        return new ChunkOffsetAndCount(offset, ChunkerUtils.countWords(offset.start(), offset.end(), wordIterator));
    }

    private List<ChunkOffsetAndCount> splitTextBySeparatorRegex(String input, ChunkOffset offset, String separatorRegex) {
        var pattern = Pattern.compile(separatorRegex, Pattern.MULTILINE);
        var matcher = pattern.matcher(input).region(offset.start(), offset.end());

        var chunkOffsets = new ArrayList<ChunkOffsetAndCount>();
        int chunkStart = offset.start();
        while (matcher.find()) {
            var chunkEnd = matcher.start();

            if (chunkStart < chunkEnd) {
                chunkOffsets.add(buildChunkOffsetAndCount(input, new ChunkOffset(chunkStart, chunkEnd)));
            }
            chunkStart = chunkEnd;
        }

        if (chunkStart < offset.end()) {
            chunkOffsets.add(buildChunkOffsetAndCount(input, new ChunkOffset(chunkStart, offset.end())));
        }

        return chunkOffsets;
    }

    private List<ChunkOffsetAndCount> mergeChunkOffsetsUpToMaxChunkSize(List<ChunkOffsetAndCount> chunkOffsets, int maxChunkSize) {
        if (chunkOffsets.size() < 2) {
            return chunkOffsets;
        }

        List<ChunkOffsetAndCount> mergedOffsetsAndCounts = new ArrayList<>();
        var mergedChunk = chunkOffsets.getFirst();
        for (int i = 1; i < chunkOffsets.size(); i++) {
            var chunkOffsetAndCountToMerge = chunkOffsets.get(i);
            var potentialMergedChunk = new ChunkOffsetAndCount(
                new ChunkOffset(mergedChunk.chunkOffset.start(), chunkOffsetAndCountToMerge.chunkOffset.end()),
                mergedChunk.wordCount + chunkOffsetAndCountToMerge.wordCount
            );
            if (isChunkWithinMaxSize(potentialMergedChunk, maxChunkSize)) {
                mergedChunk = potentialMergedChunk;
            } else {
                mergedOffsetsAndCounts.add(mergedChunk);
                mergedChunk = chunkOffsets.get(i);
            }

            if (i == chunkOffsets.size() - 1) {
                mergedOffsetsAndCounts.add(mergedChunk);
            }
        }
        return mergedOffsetsAndCounts;
    }

    private List<ChunkOffset> chunkWithBackupChunker(String input, ChunkOffset offset, int maxChunkSize) {
        var chunks = new SentenceBoundaryChunker().chunk(
            input.substring(offset.start(), offset.end()),
            new SentenceBoundaryChunkingSettings(maxChunkSize, 0)
        );
        var chunksWithOffsets = new ArrayList<ChunkOffset>();
        for (var chunk : chunks) {
            chunksWithOffsets.add(new ChunkOffset(chunk.start() + offset.start(), chunk.end() + offset.start()));
        }
        return chunksWithOffsets;
    }

    private record ChunkOffsetAndCount(ChunkOffset chunkOffset, int wordCount) {}

    /**
     * A chunk that still needs to be checked against the max chunk size, together with the index of the separator to try
     * next if it turns out to be too large.
     */
    private record PendingChunk(ChunkOffsetAndCount chunkOffsetAndCount, int separatorIndex) {}
}
