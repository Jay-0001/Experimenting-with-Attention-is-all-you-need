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
        intermediate = torch.softmax(intermediate,dim=2)             #sum across the column must add up to 1, each query with all keys
        
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
    torch.cuda.reset_peak_memory_stats()                   #Reset GPU memory usage stats
    start=time.time()                                      #Returns the time in seconds since the epoch!
    with torch.no_grad():
        attention = attn(input)
    torch.cuda.synchronize()                               #Synchronise to prevent CPU form intervening, until all GPU work is done 
    stop=time.time()
    mem=torch.cuda.max_memory_allocated()/(1024*1024)      #GPU memory allocated since the last reset
    return stop-start,mem

#Benchmarking for different sequence lengths
def performance():
    batch,seq_len,emb_dim=1,0,512
    attention_calculation=IsolatedAttention(emb_dim).to('cuda')

    x_seq,y_time=[],[]
    mem_usage=[]

    for seq_len in range(100,10001,400):
        input = torch.randn(batch,seq_len,emb_dim,device='cuda')
        bench=benchmark(input,attention_calculation)
        x_seq.append(seq_len)
        y_time.append(bench[0])
        mem_usage.append(bench[1])
        print(f"Time taken for sequence length {seq_len} is -- {bench[0]}  memory used --- {bench[1]} MB")

    plt.plot(x_seq,y_time,color='blue',label='compute time')
    plt.title('Compute time as a function of sequence length')
    plt.show()
    plt.plot(x_seq,mem_usage,color='red',label='memory usage')
    plt.title('Memory usage as a function of sequence length')
    plt.show()

performance()

'''
Tweaks
    How do I use rdtsc in Python? Is there an assembly support function like the asm?
'''

'''
Learnings
    Compute timing scales almost linearly but not quadratically in my experiments
    How does Memory scale? How can i measure the memory usage??

    Model and data must be in the GPU
    GPU is asynchronous in nature

    Understanding the growth of functions and verifying quadratic growth
    Expressing memory usage in this algorithm as a function
        (3*dm*dm+3*N^2+N*dm)
    One can learn so much more by questioning and experimenting
'''

'''
Observations
    Memory usage does not scale as drastically as I had imagined when seq_len<1000
        It does not seem quadratic wrt sequence length
    
    Verify if memory usage is quadratic using slope
    Seems like setup overhead was dominating actual memory costs
        Slight Quadratic behaviour is observed when sequence length > 1000

    Cut down Batch size to best observe sequence length
    Increase the range of sequence length to observe Quadratic behavior
    
    GPUs are memory constrained

    Why does compute time decrease right before the last iteration?
'''