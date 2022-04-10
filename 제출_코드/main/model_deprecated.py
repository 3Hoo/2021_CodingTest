##########################
# Transformer Base Model #
##########################
# 자연어 처리에 훌륭한 성능을 보이고 있는 Trasformer를 적절히 변형하여 regression을 하는 모델로 변경해보자
# 참고 : https://arxiv.org/abs/1706.03762, https://www.linkedin.com/pulse/how-i-turned-nlp-transformer-time-series-predictor-zimbres-phd, http://nlp.seas.harvard.edu/2018/04/03/attention.html

# Input은 (batch, 4, 30) 으로 30길이의 4차원 데이터
# 여기서 dx와 dy를 좀 더 제대로 구별하기 위해 DSA에서 사용한 prev differ 정보를 추가하자.
# 즉, 최종 input 데이터 shape는 (batch, 5, 30)이 될 것이다

class EncoderDecoder(nn.Module):
    '''
    기본이 되는 Enc-Dec 블록
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
class Generator(nn.Module):
    '''
    여기서, vocab size는 지금 사용하고 있는 data의 latent_dim(5)로 대체된다
    '''
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, 5)

    def forward(self, x):
        #return F.log_softmax(self.proj(x), dim=-1)
        # 그리고 transformer을 regression의 용도로 사용할 것이므로, (target이 long이 아닌 float)
        # loss function을 relu로 변경한다
        return F.relu(self.proj(x))

def clones(module, N):
    '''
    module을 N번 반복하여 return 하는 함수
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    '''
    Encoder에 사용될 각 (sub) layer에 normalization을 추가하는 모듈
    '''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    '''
    layer에 LayerNorm과 residual connection을 추가하는 모듈
    '''
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderLayer(nn.Module):
    '''
    Encoder는 self attention과 feed-forward, 두 layer 를! 쌓아서 만든다
    하나는 multi head self-attention, 또 하나는 fc layer
    그리고 두 layer 사이에 SublayerConnection 모듈을 이용하여 LayerNorm과 residual connection을 적용한다
    '''
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    '''
    Base Encoder는 encoder layer을 N번 반복한 구조
    '''
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
   
def subsequent_mask(size):
    '''
    이 마스크는 size by size square & lower triangular matrix with all elements are 1 형태이다
    '''
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 

def attention(query, key, value, mask=None, dropout=None):
    '''
    attention 함수
    return => softmax(QK^T / sqrt(dk)) * V
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    
class DecoderLayer(nn.Module):
    '''
    decoder layer는 encoder와 유사한 구조이지만, self-attention과 fc 사이에 src-attention이 추가된 구조를 가진다
    '''
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        '''
        d_model(임베딩 사이즈)와 head의 갯수를 argument로 받는다
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Decoder(nn.Module):
    '''
    Base Decoder 역시 decoder layer을 N번 반복한 구조
    단, 각 layer에 mask가 적용된다
    '''
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class PositionwiseFeedForward(nn.Module):
    '''
    encoder와 decoder의 각 layer는 fc 를 포함하고 있는데,
    이 모듈은 그 fc layer를 나타낸다
    '''
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

'''
    본래 자연어 처리에서는 정수(long)로 표현된 vocab을 입력으로 한 다음 이를 토큰화하여 임베딩레이어가 float로 변화하지만
    여기서는 데이터 input 자체가 float이므로 변형이 필요하다
'''
class Embeddings1(nn.Module):
    # 이 임베딩 레이어는 Encoder에 들어간다
    def __init__(self, d_model, vocab):
        super(Embeddings1, self).__init__()
        #self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model  # d_model = 임베딩 차원

    def forward(self, x):
        #return self.lut(x) * math.sqrt(self.d_model)
        # d_model를 4로 사용할 것이고 input 데이터의 길이는 30, latent_dim=5 이므로
        return torch.cat(4*[x]).reshape(-1, 30, self.d_model*5) * math.sqrt(self.d_model)
class Embeddings2(nn.Module):
    # 이 임베딩 레이어는 Decoder에 들어간다
    def __init__(self, d_model, vocab):
        super(Embeddings2, self).__init__()
        #self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model  # d_model = 임베딩 차원

    def forward(self, x):
        #return self.lut(x) * math.sqrt(self.d_model)
        # d_model를 4로 사용할 것이고 target 데이터의 길이는 29, latent_dim=5 이므로
        # (encoder의 input에 t-29~t초 동안의 pos변화량이 들어가면, decoder는 t-28~t+1이 pos 변화량을 출력해야 한다 )
        return torch.cat(4*[x]).reshape(-1, 29, self.d_model*5) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    '''
    임베딩 레이어 이후에 positional encoding을 추가하여 data sequence안의 토큰들 사이의 상대/절대 정보를 추가한다
    단, 이 transformer는 regression 작업을 할 것이기 때문에 positional encoding은 삭제하자
    '''
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        #x = x + Variable(self.pe[:, :x.size(1)], 
        #                 requires_grad=False)
        #return self.dropout(x)
        return x

def make_model(src_vocab, tgt_vocab, N=2, 
               d_model=4, d_ff=32, h=4, dropout=0.1):
    '''
    최종적으로 Transformer 모델을 만드는 함수
    여기서는 너무 무겁지 않도록 하이퍼파라미터를 낮게 조정했다
    '''
    d_model = d_model * 5
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings1(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings2(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model