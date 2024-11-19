import pickle

class Vocab: 
    def __init__(self, special_tokens=None):
        """
        Inicializa o vocabulário.
        :param special_tokens: Lista de tokens especiais como <pad>, <start>, <end>, <unk>.
        """
        self.word2idx = {}
        self.idx2word = {}
        self.freqs = {}  # Frequências das palavras
        self.special_tokens = special_tokens if special_tokens else ['<pad>', '<start>', '<end>', '<unk>']
        
        # Adiciona tokens especiais ao vocabulário
        for token in self.special_tokens:
            self.add_word(token)

    def add_word(self, word):
        """
        Adiciona uma palavra ao vocabulário.
        :param word: Palavra a ser adicionada.
        """
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.freqs[word] = 1
        else:
            self.freqs[word] += 1

    def build_vocab(self, sentences, min_freq=1):
        """
        Constrói o vocabulário a partir de uma lista de sentenças.
        :param sentences: Lista de strings (ex.: ['uma frase', 'outra frase']).
        :param min_freq: Frequência mínima para incluir uma palavra no vocabulário.
        """
        for sentence in sentences:
            for word in sentence.split():
                self.add_word(word)
        
        # Remove palavras com frequência menor que o limite
        self.word2idx = {word: idx for word, idx in self.word2idx.items() if self.freqs[word] >= min_freq or word in self.special_tokens}
        
        # Atualiza idx2word com as palavras que restaram
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Remove palavras com baixa frequência
        self.freqs = {word: freq for word, freq in self.freqs.items() if word in self.word2idx}

    def numericalize(self, sentence):
        """
        Converte uma sentença em uma lista de índices.
        :param sentence: String (ex.: "uma frase").
        :return: Lista de índices (ex.: [1, 5, 10]).
        """
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence.split()]

    def decode(self, indices):
        """
        Converte uma lista de índices em uma sentença.
        :param indices: Lista de índices (ex.: [1, 5, 10]).
        :return: String (ex.: "uma frase").
        """
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices])

    def get_frequency(self, word):
        """
        Retorna a frequência de uma palavra no vocabulário.
        :param word: Palavra a ser verificada.
        :return: Frequência da palavra.
        """
        return self.freqs.get(word, 0)

    def save(self, filename):
        """
        Salva o vocabulário em um arquivo.
        :param filename: Caminho do arquivo onde o vocabulário será salvo.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Carrega um vocabulário de um arquivo.
        :param filename: Caminho do arquivo do vocabulário.
        :return: Instância do vocabulário.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        """
        Retorna o tamanho do vocabulário.
        """
        return len(self.word2idx)
