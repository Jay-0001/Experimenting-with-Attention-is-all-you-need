#Implementing multi headed attention to leverage the true parallelism provided by this paper
import torch 
import time
import matplotlib.pyplot as plt


#Inputs -- sets of queries, keys and values in the form of matrices -- dimension (batch x seq x dmodel)
    #They're all linear projections of the same input embedding btw
#Linearly projected matrices are broken down to calculate attention in a parallelized manner across diff heads
#Output values are concatenated across diff heads and finally projected once again
#Outputs -- (batch x seq x dmodel) after attention calculations
#How to break down the Attention expression into implementable steps

class IsolatedMultiAttention(torch.nn.Module):
    #Extending nn.module allows the class to store params, handle gradients
    def __init__(self, d_model=512,heads=8):
        super().__init__()
        self.d_model=d_model
        #The matrix multiplications must result in batch x seq x d_model
        self.Weight_Q=torch.nn.Linear(d_model,d_model,bias=False)
        self.Weight_K=torch.nn.Linear(d_model,d_model,bias=False)
        self.Weight_V=torch.nn.Linear(d_model,d_model,bias=False)
        self.num_heads=heads
        #num_heads*dv == d_model
        self.Weight_O=torch.nn.Linear(d_model,d_model,bias=False)

    #the mythical abstractor -- forward() ; activates when passing values directly to the class's object
    #Multi Headed Attention version -- 8 heads as choosen in the paper
    def forward(self, inp_emb):
        #linear projection of input into Q,K,V matrices
        Q=self.Weight_Q(inp_emb)
        K=self.Weight_K(inp_emb)
        V=self.Weight_V(inp_emb)

        #Break down the linearly projected matrices into num_heads
        assert self.d_model%self.num_heads == 0                              #Ensuring consistency in dimensions
        d_k = self.d_model//self.num_heads
        Q_heads = torch.split(Q,split_size_or_sections=d_k,dim=2)  #Tuple of broken down tensor
        K_heads = torch.split(K,split_size_or_sections=d_k,dim=2)
        V_heads = torch.split(V,split_size_or_sections=d_k,dim=2)

        concat_output=[]
        for i in range(self.num_heads):
            #matrix multiplication between Q and K^t
            K_t = K_heads[i].transpose(1,2)                                  # Dimensions to be considered for the transpose operation
            intermediate = torch.matmul(Q_heads[i],K_t)                      # launches a CUDA kernel
            
            #scale wrt to the heads                                   
            intermediate = intermediate/(d_k**0.5)

            #Softmax operation
            intermediate = torch.softmax(intermediate,dim=2)             #sum across the column must add up to 1, each query with all keys
        
            #Weighing the V vector with the calculated compatibility
            attention = intermediate @ V_heads[i]
            concat_output.append(attention)

        #Concatenate output across all heads
        final_output= torch.cat(concat_output,dim=2)

        #The final linear projection
        final_output=self.Weight_O(final_output)
        return final_output

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
    attention_calculation=IsolatedMultiAttention(emb_dim,2).to('cuda')

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
    Using Assert statements to verify dimensions
'''

'''
Learnings
    How is parallel computation done in the GPU?
        Each core doing a part of the task at the same  time!!

    What exactly is a linear projection?
        How can matrix multiplication considered as projection?

    How to parallelize execution?
        Should I be adding every single view to the GPU or just the original tensor?
        An extra dimension instead of a loop??

    Which tensors to send to the cuda?
        Only the model and the original tensor
        Derived tensors inherit the original tensor's device

    Always dry run and predict output before execution
'''

'''
Observations
    Does this version use parallelization?
        It does, just not optimally
        Spawning indivual matrix multiplication kernels 

    Multi Headed Attention is performing way worser than the single headed version!!
        Something's terribly wrong with my parallelization 
        Similar memory usage as the single headed version but much more compute time
        What is increasing my compute time

    The kernel overhead
        GPU does not simply parallelize a loop
    
    Understanding why memory gets exhausted
        Sequential allocation for H times
'''