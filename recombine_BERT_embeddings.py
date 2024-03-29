import json
import h5py
import numpy as np
from hedgepig_logger import log

def readOverlaps(f):
    overlaps = []
    cur_line_overlaps = []
    with open(f, 'r') as stream:
        for line in stream:
            if len(line.strip()) > 0:
                cur_line_overlaps.append(int(line))
            else:
                if len(cur_line_overlaps) > 0:
                    overlaps.append(cur_line_overlaps)
                    cur_line_overlaps = []

    if len(cur_line_overlaps) > 0:
        overlaps.append(cur_line_overlaps)

    return overlaps

def streamingBERTConvert(bertf, overlaps, outf, tokenizedf):
    line_index = 0
    overlap_index = 0

    log.track('  >> Processed {0:,} lines of BERT output ({1:,}/%s text lines)' % (
        '{0:,}'.format(len(overlaps))
    ))
    line_embeddings_by_layer = {}
    with open(bertf, 'r') as bert_stream, \
         h5py.File(outf, 'w') as h5stream, \
         open(tokenizedf, 'w') as token_stream:
        for line in bert_stream:
            data = json.loads(line)
            all_tokens = data['features']
            # blank lines in the input still get entered in JSON;
            # skip output from those lines to maintain proper alignment
            if len(all_tokens) > 0:
                for i in range(len(all_tokens)):
                    token_stream.write(all_tokens[i]['token'])
                    if i < len(all_tokens)-1:
                        token_stream.write(' ')
                    else:
                        token_stream.write('\n')

                overlap_quantity = overlaps[line_index][overlap_index]
                for token_embedding in all_tokens[:len(all_tokens)-overlap_quantity]:
                    for embedding_layer in token_embedding['layers']:
                        layer_ix = embedding_layer['index']
                        if not layer_ix in line_embeddings_by_layer:
                            line_embeddings_by_layer[layer_ix] = []
                        line_embeddings_by_layer[layer_ix].append(embedding_layer['values'])

                if overlap_quantity == 0:
                    # hit end of line, so construct numpy tensor as
                    # [ <layer>, <token_ix>, <values> ]
                    line_tensor = np.array([
                        layer_token_values
                            for (layer_ix, layer_token_values)
                            in sorted(
                                line_embeddings_by_layer.items(),
                                key=lambda pair: pair[0]
                            )
                    ])
                    h5stream.create_dataset(
                        str(line_index),
                        data=line_tensor
                    )
                    line_index += 1
                    overlap_index = 0
                    line_embeddings_by_layer = {}

                else:
                    overlap_index += 1

            log.tick(line_index+1)
    log.flushTracker(line_index)

if __name__ == '__main__':
    def _cli():
        import optparse
        parser = optparse.OptionParser(usage='Usage: %prog')
        parser.add_option('--bert-output', dest='bert_f',
            help='(REQUIRED) BERT embeddings file (JSON)')
        parser.add_option('--overlaps', dest='overlaps_f',
            help='(REQUIRED) file indicating line-to-line overlaps')
        parser.add_option('-o', '--output', dest='output_f',
            help='(REQUIRED) .hdf5 file to write output to')
        parser.add_option('--tokenized', dest='tokenized_f',
            help='(REQUIRED) file to write BERT tokenization to')
        parser.add_option('-l', '--logfile', dest='logfile',
            help='name of file to write log contents to (empty for stdout)',
            default=None)

        (options, args) = parser.parse_args()

        if not options.bert_f:
            parser.error('Must provide --bert-output')
        elif not options.overlaps_f:
            parser.error('Must provide --overlaps')
        elif not options.output_f:
            parser.error('Must provide --output')

        return options
    options = _cli()
    log.start(options.logfile)
    log.writeConfig([
        ('BERT output', options.bert_f),
        ('Overlaps file', options.overlaps_f),
        ('Output file', options.output_f),
    ], 'BERT embedding recombination')

    log.writeln('Reading overlaps from %s...' % options.overlaps_f)
    overlaps = readOverlaps(options.overlaps_f)
    log.writeln('Read overlaps for {0:,} lines.\n'.format(len(overlaps)))

    log.writeln('Streaming BERT output conversion...')
    streamingBERTConvert(
        options.bert_f,
        overlaps,
        options.output_f,
        options.tokenized_f
    )
    log.writeln('Done.')

    log.stop()
