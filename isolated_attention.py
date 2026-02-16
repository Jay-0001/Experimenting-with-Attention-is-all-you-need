#Isolating the scaled dot product attention used in "Attention is all you need"
import torch 
import time
import matplotlib.pyplot as plt


#Inputs -- sets of queries, keys and values in the form of matrices -- dimension (seq x dmodel)
    #They're all linear projections of the same input embedding btw
#The attention algorithm actually performs two tasks -- similarity calculation (Q,K) & information transfer(V)
#Outputs -- (seq x dmodel) after attention calculations
#How to break down the Attention expression into implementable steps

class IsolatedAttention(torch.nn.Module):
    #Extending nn.module allows the class to store params, handle gradients
    def __init__(self, d_model):
        super().__init__()
        self.d_model=d_model
        #The matrix multiplications must result in batch x seq x d_model
        self.Weight_Q=torch.nn.Linear(d_model,d_model,bias=False)
        self.Weight_K=torch.nn.Linear(d_model,d_model,bias=False)
        self.Weight_V=torch.nn.Linear(d_model,d_model,bias=False)

    #the mythical abstractor -- forward() ; activates when passing values directly to the class's object
    #can visualize this function as the first sublayer in the encodere
    def forward(self, inp_emb):
        #linear projection of input into Q,K,V matrices
        Q=self.Weight_Q(inp_emb)
        K=self.Weight_K(inp_emb)
        V=self.Weight_V(inp_emb)

        #matrix multiplication between Q and K^t
        K_t = K.transpose(1,2)                                  # Dimensions to be considered for the transpose operation
        intermediate = torch.matmul(Q,K_t)                                     # Has interesting behaviour based on the params
        intermediate = intermediate/(self.d_model**0.5)
        
        #Softmax operation
        intermediate = torch.softmax(intermediate,dim=2)             #sum across the column must add up to 1, each query with all keys --  pairwise logic does not add up, check on it!!
        
        #Weighing the V vector with the calculated compatibility
        attention = intermediate @ V

        return attention
    
#Can I visualize this entire class as a NN with three independent layers?
#I need more clarity on what's possible with the forward function

#Set the params here!
'''
batch,seq_len,emb_dim=10,20,100
input = torch.randn(batch,seq_len,emb_dim)
attention_calculation=IsolatedAttention(emb_dim)
attention=attention_calculation(input)
print(attention.shape)                                  #Dimensionality flows as intended
'''
                                        
def benchmark(input,attn):
    #warming up :? 
    for _ in range(3):
        attn(input)
    
    #benchmarking!  -- easiest time to use fr
    start=time.time()                                      #Returns the time in seconds since the epoch!
    with torch.no_grad():
        attention = attn(input)
    stop=time.time()

    return stop-start

#Benchmarking for different sequence lengths
def performance():
    batch,seq_len,emb_dim=1,0,512
    attention_calculation=IsolatedAttention(emb_dim)

    x_seq,y_time=[],[]

    for seq_len in range(100,1001,100):
        input = torch.randn(batch,seq_len,emb_dim)
        time_taken=benchmark(input,attention_calculation)
        x_seq.append(seq_len)
        y_time.append(time_taken)
        print(f"Time taken for sequence length {seq_len} is -- {time_taken}")

    plt.plot(x_seq,y_time)
    plt.title('visualizing time taken as a function of sequence length')
    plt.show()

performance()

'''
Tweaks
    Try adding assert statements to verify the dimensionality
    Play around with tensors
    What are the other inheretible classes like the nn.module
'''

'''
Learnings
    Revisiting PyTorch fundamentals
    Understanding the significance of learned linear projections over raw embeddings
        Learnability completely changes what can achieved with a parameter
    I might have to implement the multi headed attention to leverage the parallelism provided by this paper
'''

'''
Observations
    sequence lengths of 100 and 500 returned 0 when i attempted 10 warm up iterations
    Embedding size changed to 512 to match the paper's d_model size
    As the sequence length increases by a factor of 5, the time taken increases by a factor of 10
'''