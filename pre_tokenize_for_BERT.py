'''
Convert pre-tokenized data to BERT tokenization for alignment
'''

import bert.tokenization
import numpy as np
from hedgepig_logger import log

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('-i', '--input', dest='input_file',
            help='(REQUIRED) input file to tokenize with BERT')
        parser.add_option('-o', '--output', dest='output_file',
            help='(REQUIRED) base path for output files; creates .tokens'
                 ' file with BERT tokenization but no subsequence splitting,'
                 ' .subsequences file with subsequence splitting for BERT max'
                 ' length, .overlaps file tracking subsequence splits, and'
                 ' .log file for logging output.')
        parser.add_option('-s', '--max-sequence-length', dest='max_sequence_length',
            type='int', default=128,
            help='maximum subsequence length for BERT embedding (default %default;'
                 ' note that actual subsequence length used will be this -2, to'
                 ' account for special [CLS] and [SEP] tokens for BERT processing)')
        parser.add_option('--overlap', dest='overlap',
            type='float', default=0.5,
            help='portion of split subsequences to overlap for averaging (default %default)')
        parser.add_option('-v', '--vocab-file', dest='vocab_file',
            help='(REQUIRED) BERT vocab file for tokenization')
        (options, args) = parser.parse_args()

        if not options.input_file:
            parser.error('Must provide --input')
        elif not options.output_file:
            parser.error('Must provide --output')
        elif not options.vocab_file:
            parser.error('Must provide --vocab-file')
        elif options.overlap < 0 or options.overlap >= 1:
            parser.error('--overlap must be between [0,1)')

        return options
    options = _cli()

    output_tokens = '%s.tokens' % options.output_file
    output_subsequences = '%s.subsequences' % options.output_file
    output_overlaps = '%s.overlaps' % options.output_file
    output_log = '%s.log' % options.output_file

    log.start(output_log)
    log.writeConfig([
        ('Input file', options.input_file),
        ('Output settings', [
            ('Base path', options.output_file),
            ('Tokenized file', output_tokens),
            ('Subsequences file', output_subsequences),
            ('Overlaps file', output_overlaps),
            ('Log file', output_log),
        ]),
        ('Max subsequence length', options.max_sequence_length),
        ('Overlap fraction', options.overlap),
        ('BERT vocab file', options.vocab_file)
    ])

    options.max_sequence_length -= 2

    log.writeln('Tokenizing input file %s...' % options.input_file)
    tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=options.vocab_file,
        do_lower_case=True
    )
    num_lines = 0
    with open(options.input_file, 'r') as input_stream, \
         open(output_tokens, 'w') as output_stream:
        for line in input_stream:
            tokens = tokenizer.tokenize(line.strip())
            output_stream.write('%s\n' % (' '.join(tokens)))
            num_lines += 1
    log.writeln('Wrote {0:,} tokenized lines.\n'.format(num_lines))

    log.writeln('Reading tokenized lines from %s...' % output_tokens)
    tokenized_lines = []
    with open(output_tokens, 'r') as stream:
        for line in stream:
            tokenized_lines.append(line.strip().split())
    log.writeln('Read {0:,} tokenized lines.\n'.format(len(tokenized_lines)))

    log.writeln('Splitting into subsequences for BERT processing...')
    with open(output_subsequences, 'w') as stream, \
         open(output_overlaps, 'w') as overlap_stream:
        for input_line in tokenized_lines:
            cur_start = 0
            while cur_start < len(input_line):
                next_sequence = ['[CLS]']
                next_sequence.extend(input_line[cur_start:cur_start + options.max_sequence_length])
                next_sequence.append('[SEP]')
                stream.write('%s\n' % (' '.join(next_sequence)))

                if cur_start + options.max_sequence_length < len(input_line):
                    real_overlap = options.overlap * options.max_sequence_length
                    int_overlap = int(real_overlap)
                    cur_start += (options.max_sequence_length - int_overlap)
                    overlap_stream.write('%d\n' % int_overlap)
                else:
                    cur_start += options.max_sequence_length
                    overlap_stream.write('0\n')
            stream.write('\n')
            overlap_stream.write('\n')
    log.writeln('  Wrote split subsequences to %s' % output_subsequences)
    log.writeln('  Wrote overlap info to %s\n' % output_overlaps)

    log.stop()
