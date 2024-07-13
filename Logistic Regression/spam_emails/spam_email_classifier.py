import re
from collections import Counter

def get_features(email_contents: str) -> list[float]:
    # Target data
    column_names = [
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
        'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
        'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
        'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
        'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
        'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
        'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
        'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
        'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
        'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
        'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
        'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
        'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
        'char_freq_$', 'char_freq_#', 'capital_run_length_average',
        'capital_run_length_longest', 'capital_run_length_total', 'is_spam'
    ]
    characters = Counter(email_contents)
    words = Counter(re.findall(r'\w+', email_contents))

    uninterupted_sequence_of_capital = re.findall(r'\b[A-Z]{2,}', email_contents)
    if not uninterupted_sequence_of_capital:
        capital_run_length_average = 0
        capital_run_length_longest = 0
    else:
        capital_run_length_average = sum(len(word) for word in uninterupted_sequence_of_capital) / len(uninterupted_sequence_of_capital)
        capital_run_length_longest = max(len(word) for word in uninterupted_sequence_of_capital)
    capital_run_length_total = sum(len(capital) for capital in re.findall(r'[A-Z]', email_contents))

    features = [
        word_freq(words, 'make'), word_freq(words, 'address'), word_freq(words, 'all'), word_freq(words, '3d'),
        word_freq(words, 'our'), word_freq(words, 'over'), word_freq(words, 'remove'), word_freq(words, 'internet'),
        word_freq(words, 'order'), word_freq(words, 'mail'), word_freq(words, 'receive'), word_freq(words, 'will'),
        word_freq(words, 'people'), word_freq(words, 'report'), word_freq(words, 'addresses'), word_freq(words, 'free'),
        word_freq(words, 'business'), word_freq(words, 'email'), word_freq(words, 'you'), word_freq(words, 'credit'),
        word_freq(words, 'your'), word_freq(words, 'font'), word_freq(words, '000'), word_freq(words, 'money'),
        word_freq(words, 'hp'), word_freq(words, 'hpl'), word_freq(words, 'george'), word_freq(words, '650'),
        word_freq(words, 'lab'), word_freq(words, 'labs'), word_freq(words, 'telnet'), word_freq(words, '857'),
        word_freq(words, 'data'), word_freq(words, '415'), word_freq(words, '85'), word_freq(words, 'technology'),
        word_freq(words, '1999'), word_freq(words, 'parts'), word_freq(words, 'pm'), word_freq(words, 'direct'),
        word_freq(words, 'cs'), word_freq(words, 'meeting'), word_freq(words, 'original'), word_freq(words, 'project'),
        word_freq(words, 're'), word_freq(words, 'edu'), word_freq(words, 'table'), word_freq(words, 'conference'),
        char_freq(characters, ';'), char_freq(characters, '('), char_freq(characters, '['), char_freq(characters, '!'),
        char_freq(characters, '$'), char_freq(characters, '#'), capital_run_length_average,
        capital_run_length_longest, capital_run_length_total
    ]

    return features


def word_freq(words: Counter, word: str) -> float:
    return 100 * words.get(word, 0) / sum(words.values())


def char_freq(characters: Counter, char: str) -> float:
    return 100 * characters.get(char, 0) / sum(characters.values())


if __name__ == '__main__':
    email_contents = ('Good Evening, For now ignore those four trouble makers (including Scientific American since if '
                      'they have a paywall for most articlesm readers when directed there will be a bit annoyed that '
                      'they can’t finish reading the article). We already are getting a lot of content and while '
                      'those are good sources, they are not indispensable. If Ben Miller is unresponsive you can '
                      'remove him. Just let me know so that I can get him off the github and our own record books. '
                      'Respectfully, Amitabha')
    get_features(email_contents)