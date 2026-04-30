def relu(x):
    return max(0,x)
# 负输入来讲，直接输出0，导致有用的神经元”死亡“
# 如果,有用的神经元over了,信息流动确实or不灵活
# 在超大scale下的训练,这种大概率导致神经元over的激活函数,积少成多,导致很多的message学不到了


# 引入了swiglu:
# swish + glu(门控线性单元)
# swish:
def swish(x,beta=1.0):
    return x * sigmoid(beta*x)

# glu
def glu(x,w,v,b,c):
    a = x @ w + b
    b = x @ v + c
    return a * sigmoid(b) #门控机制

# swish+glu 就是一个套娃
def swiglu(x,w,v,b,c):
    a = x @ w + b
    b = x @ v + c
    return a * swish(b)

# transformer里的ffn
class traditionalffn(nn.Module):
    def _init_(self,d_model,d_ff):
        self.L1 = nn.linear(d_model,d_ff)
        self.active = nn.relu()
        self.L2 = nn.linear(d_ff,d_model)
    def forward(self,x):
        return self.L2(self.active(self.L1(x)))
    
# 在LLama里面的实现
class swigluffn(nn.Module):
    def _init_(self,d_model,d_ff):
        # 此时,三个线性层,不是两个了
        self.w1 = nn.linear(d_model,d_ff) #用于门控
        self.w2 = nn.linear(d_model,d_ff) #用于门控
        self.w3 = nn.linear(d_ff,d_model) #用于输出投影
    def forward (self,x):
        gate = self.w1(x)#门控信号
        value = self.w2(x)
        gated = gate * F.silu(value) #swish激活
        return self.w3(gated)


