TEMPLATES = [
    "{} Using only the above description and what you know about the world, ", '"{}" is definitely correct. Yes or no? <mask>',
    "{}", "\nquestion: {}Yes or no?\nanswer:<mask>",
    "{}", ' Are we justified in saying that "{}"?<mask>',
    "Given {}", ' Should we assume that "{}" is true?<mask>',
    "{} Based on the previous passage, ", 'is it true that {}, "?<mask>',
    "Given {}", ' Is it guaranteed true that "{}"?<mask>',
    'Suppose {}', ' Can we infer that "{}"?<mask>',
    'Given that {}', 'Does it follow that "{}"?<mask>',
    "{}", ' Question: Does this imply that "{}"?<mask>',
    'Given that {}', ' Therefore, it must be true that "{}"?<mask>',
    "{}", ' Do most of the above words appear in the following passage? {}<mask>']

# ground truth + mask
TEMPLATES = [
    '{} Using only the above description and what you know about the world, "" is definitely correct. Yes or no?', '{}',
    "{}\nquestion: Yes or no?\n", "answer:{}",
    '{} Are we justified in saying that ""?', '{}',
    'Given {} Should we assume that "" is true?', '{}',
    '{} Based on the previous passage, is it true that , "?', '{}',
    'Given {} Is it guaranteed true that ""?', '{}',
    'Suppose {} Can we infer that ""?', '{}',
    'Given that {}Does it follow that ""?', '{}',
    '{} Question: Does this imply that ""?', '{}',
    'Given that {} Therefore, it must be true that ""?', '{}',
    "{} Do most of the above words appear in the following passage? ", '{}']

# ground truth + question + mask
TEMPLATES = [
    '{} Using only the above description and what you know about the world, "{}" is definitely correct. Yes or no?', '{}',
    "{}\nquestion: {}Yes or no?\n", "answer:{}",
    '{} Are we justified in saying that "{}"?', '{}',
    'Given {} Should we assume that "{}" is true?', '{}',
    '{} Based on the previous passage, is it true that {}, "?', '{}',
    'Given {} Is it guaranteed true that "{}"?', '{}',
    'Suppose {} Can we infer that "{}"?', '{}',
    'Given that {}Does it follow that "{}"?', '{}',
    '{} Question: Does this imply that "{}"?', '{}',
    'Given that {} Therefore, it must be true that "{}"?', '{}',
    "{} Do most of the above words appear in the following passage? {}", '{}']

# question + mask
TEMPLATES = [
    (' Using only the above description and what you know about the world, "{}" is definitely correct. Yes or no?', '{}',
    ("\nquestion: {}Yes or no?\n", "answer:{}",
    (' Are we justified in saying that "{}"?', '{}',
    ('Given  Should we assume that "{}" is true?', '{}',
    (' Based on the previous passage, is it true that {}, "?', '{}',
    ('Given  Is it guaranteed true that "{}"?', '{}',
    ('Suppose  Can we infer that "{}"?', '{}',
    ('Given that Does it follow that "{}"?', '{}',
    (' Question: Does this imply that "{}"?', '{}',
    ('Given that  Therefore, it must be true that "{}"?', '{}',
    (" Do most of the above words appear in the following passage? {}", '{}']
            # ("{}", [' Are there lots of similar words in ', '"', text_b, '"?', self.mask],
            # ("{}", [' Does that have the same meaning as ', '"', text_b, '"?', self.mask],
            # ("{}", [' Can that be paraphrased as: ', '"', text_b, '"?', self.mask],
            # ("{}", [' Can that be summarized as ', '"', text_b, '"?', self.mask],
            # ("{}", [' Does the paragraph start with "the"? ', text_b, self.mask],
            # ("{}", [' Is this grammatically correct? ', text_b, self.mask],
            # ("{}", [' Is the sentiment positive? ', text_b, self.mask],
            # ("{}", [' Is this a sports news? ', text_b, self.mask],
            # ("{}", [' Is this French? ', text_b, self.mask],
            # ("{}", [' Single-family zoning is bad for American cities. ', '"', text_b, '"?', self.mask],
            # ("{}", [' When Bolyai sent Gauss his discovery of non-Euclidean geometry, Gauss replied that he arrived at the same results 30 years ago. "', text_b, '"?', self.mask],
            # ("{}", [' If bonito flakes boil more than a few seconds, the stock becomes too strong? "', text_b, '"?', self.mask],
            # ("{}", [' Inflections are annoying and thank god that Middle English got rid of most of them. "', text_b, '"?', self.mask],
            # ("{}", [' Is the pious loved by the gods because it is pious? Or is it pious because it is loved by the gods? ', text_b, '"?', self.mask],
            # ([text_b], [self.mask, text_a],
            # ("{}", [self.mask, text_b]}