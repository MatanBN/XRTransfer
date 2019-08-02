import sys

from tree import fill_govern_tag, op_tags, find_pot_aspects, noun_tags, count_node_aspects, decompse_sent
from utils import read_file_lines, convert_lines_to_trees, write_file_lines, Aspect

# Decompose sentences with aspects by the aspects as pivots to decompose the sentence.
if __name__ == "__main__":
    files = ['data/semeval15/train', 'data/semeval15/dev', 'data/semeval15/test',
             'data/semeval16/train', 'data/semeval16/dev', 'data/semeval16/test']
    for file in files:
        lines = read_file_lines(file + "_parse")
        sents = list()
        for line in lines:
            line_splitted = line.split(u'||||')
            tree = convert_lines_to_trees([line_splitted[0]])[0]

            aspects = list()
            for aspect in line_splitted[1:]:
                aspect_splitted = aspect.split(u'<->')
                aspects.append(Aspect(None, aspect_splitted[0], aspect_splitted[1]))
            # Count the number of aspects governed by each node.
            count_node_aspects(tree, aspects)
            """
            Find which nodes govern opinions (i.e. a node governs an opinion if one of its
            leafs decendants contains an adjective or a verb).
            """
            fill_govern_tag(tree, op_tags, aspects)
            # Decompose the sentence to several fragments, where each fragment is linked to an aspect.
            decompse_sent(tree, aspects)
            label = tree.label
            sent_string = tree.phrase + u"<->6"
            sents_parts = set()
            # Write the sentence with its label and its fragments.
            for asp in aspects:
                sent_string += u"||||" + asp.linked_node.phrase + u"<->" + asp.expression_term + u"<->" + str(asp.gold_sentiment)
            sents.append(sent_string)

        write_file_lines(file, u"".join(sents))