from tree import fill_govern_tag, op_tags, find_pot_aspects, noun_tags, count_node_aspects, decompse_sent
from utils import read_file_lines, convert_lines_to_trees, write_file_lines
# Decompose sentences without aspects by using nouns as pivots to decompose the sentence.
if __name__ == "__main__":
    lines = read_file_lines("data/parsed")
    sents = list()
    for i in range(0, len(lines), 1000):
        print str(i) + " Sentences Decomposed"
        # Convert trees in string format to trees in object format.
        trees = convert_lines_to_trees(lines[i:i+1000])
        for tree in trees:
            """
            Find which nodes govern opinions (i.e. a node governs an opinion if one of its
            leafs decendants contains an adjective or a verb).
            """
            fill_govern_tag(tree, op_tags)
            aspects = list()
            # Add all nouns to a list.
            find_pot_aspects(tree, aspects, noun_tags)
            # Count the number of aspects (nouns in this case), governed by each node.
            count_node_aspects(tree, aspects)
            # Decompose the sentence to several fragments.
            decompse_sent(tree, aspects)
            label = tree.label
            sent_string = tree.phrase + u"<->6"
            sents_parts = set()
            # Write the sentence with its label and its fragments.
            for asp in aspects:
                if asp.linked_node is not None and asp.linked_node.papa is not None:
                    sents_parts.add(asp.linked_node.phrase)
            for part in sents_parts:
                sent_string += u"||||" + part + u"<->NULL<->6"
            sents.append(sent_string)

    write_file_lines("data/unlabeled", u"\n".join(sents))