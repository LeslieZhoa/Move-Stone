## Basic Theory for NLP
### N-Gram Model
1. 举例2-gram language model:<br>
    首先假设各特征相互独立<br>
<img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20p%28w_1%2Cw_2%2Cw_3%2C...%2Cw_n%29%26%3Dp%28w_1%29%5Ccdot%20p%28w_2%7Cw_1%29%20%5Ccdot%20p%28w_3%7Cw_1%2Cw_2%29%20%5Ccdot%20%5C%20...%20%5C%20%5Ccdot%20p%28w_n%7Cw_1%2Cw_2%2C...%2Cw_n-1%29%20%5C%5C%5B2ex%5D%20%26%20%5Capprox%20p%28w_1%29%20%5Ccdot%20p%28w_2%7Cw_1%29%20%5Ccdot%20p%28w_3%7Cw_2%29%5Ccdot%20%5C%20...%20%5C%20%5Ccdot%20p%28w&plus;n%7Cw_n-1%29%20%5Cend%7Baligned%7D"/><br>
为使模型平滑化，加入back-off-->往前多看几眼<br>
例：线性插值：<img src="http://latex.codecogs.com/gif.latex?p%28w_n%7Cw_%7Bn-2%7D%2Cw_%7Bn-1%7D%29%3D%5Clambda_3%20%5C%20p%28w_n%7Cw_%7Bn-2%7D%2Cw_%7Bn-1%7D%29&plus;%5Clambda_2%20%5C%20p%28w_n%7Cw_n-1%29&plus;%5Clambda_1%20%5C%20p%28w_n%29%20%5C%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5Clambda_1%20&plus;%20%5Clambda_2&plus;%5Clambda_3%3D1"/>
2. N-Granm在模糊匹配中的应用:<br>
    N-Gram距离：<img src="http://latex.codecogs.com/gif.latex?%7CG_N%28s%29%7C&plus;%7CG_N%28t%29%7C-2%20%5Ctimes%20%7CG_N%28s%29%5Ccap%20G_N%28t%29%7C"/><br>
例：Gorbachev和Gorbechyov，N=2<br>
&nbsp; &nbsp; &nbsp; &nbsp; <u>Go</u>,<u>or</u>,<u>rb</u>,ba,ac,<u>ch</u>,he,ev<br>
&nbsp; &nbsp; &nbsp; &nbsp; <u>Go</u>,<u>or</u>,<u>rb</u>,be,<u>ch</u>,hy,yo,ov<br>
距离=8+9-2x4=9
### Recurrent Neural Network Language Models
- 公式：<br>
    <img src="http://latex.codecogs.com/gif.latex?h_n%3Dg%28V%5Bx_n%3Bh_%7Bn-1%7D%5D&plus;c%29%20%5C%5C%5B2ex%5D%20%5C%20%5Chat%20p_n%3Dsoftmax%28wh_n&plus;b%29"/>
- 反向传播：<br>
<img src="http://latex.codecogs.com/gif.latex?h_n%3Dg%28V_xx_n&plus;V_hh_%7Bn-1%7D&plus;c%29%20%5C%5C%5B2ex%5D%20%5C%20V_xx_n&plus;V_hh_%7Bn-1%7D&plus;c%3D%20z_n%20%5C%5C%5B2ex%5D%20%5C%20%5Cfrac%7B%5Cpartial%20h_n%7D%7B%5Cpartial%20z_n%7D%3Ddiag%28g%27%28z_n%29%29%20%5C%5C%5B2ex%5D%20%5C%20%5Cfrac%7B%5Cpartial%20z_n%7D%7B%5Cpartial%20h_n-1%7D%3DV_h%20%5C%5C"/><br>
<img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Cfrac%7B%5Cpartial%20costN%7D%7B%5Cpartial%20h_1%7D%20%26%3D%20%5Cfrac%7B%5Cpartial%20costN%7D%7B%5Cpartial%20%5Chat%20p_n%7D%5Ccdot%20%5Cfrac%7B%5Cpartial%20%5Chat%20p_n%7D%7B%5Cpartial%20h_n%7D%5Ccdot%20%28%5Cprod_%7Bn%20%5Cin%20%5C%7BN%2C...%2C2%5C%7D%7D%5Cfrac%7B%5Cpartial%20h_n%7D%7B%5Cpartial%20h_%7Bn-1%7D%7D%5Ccdot%20%5Cfrac%7B%5Cpartial%20z_n%7D%7B%5Cpartial%20h_%7Bn-1%7D%7D%29%20%5C%5C%5B2ex%5D%20%26%3D%20%5Cfrac%7B%5Cpartial%20costN%7D%7B%5Cpartial%20%5Chat%20p_n%7D%5Ccdot%20%5Cfrac%7B%5Cpartial%20%5Chat%20p_n%7D%7B%5Cpartial%20h_n%7D%5Ccdot%20%28%5Cprod_%7Bn%20%5Cin%20%5C%7BN%2C...%2C2%5C%7D%7Ddiag%28g%27%28z_n%29V_h%29%20%5Cend%7Baligned%7D"/><br>
  - 如果Vh最大特征值大于1，累乘之后可能出现梯度爆炸
  - 如果Vh最大特征值小于1，累乘之后可能出现梯度消失<br>
### Long Short Term Memory
&nbsp; &nbsp; 为解决梯度问题，改变结构，加入输入门和遗忘门<br>
- 公式：<br>
<img src="http://latex.codecogs.com/gif.latex?f_t%3D%5Csigma%28W_fx_t&plus;U_fh_%7Bt-1%7D&plus;b_f%29--%3Eforget%20%5C%20%5C%20gate%20%5C%5C%5B2ex%5D%20%5C%20i_t%3D%5Csigma%28W_ix_t&plus;U_ih_%7Bt-1%7D&plus;b_i%29--%3Einput%20%5C%20%5C%20gate%20%5C%5C%5B2ex%5D%20%5C%20o_t%3D%5Csigma%28W_ox_t&plus;U_oh_%7Bt-1%7D&plus;b_o%29--%3Eoutput%20%5C%20%5C%20gate%20%5C%5C%5B2ex%5D%20%5Chat%20c_t%20%3D%20%5Ctanh%28W_cx_t&plus;U_ch_%7Bt-1%7D&plus;b_c%29%20%5C%5C%5B2ex%5D%20c_t%20%3D%20f_t%20%5Ccirc%20c_%7Bt-1%7D&plus;i_t%5Ccirc%20%5Chat%20c_t%20%5C%5C%5B2ex%5D%20h_t%20%3D%20o_t%20%5Ccirc%20%5Ctanh%28c_t%29"/>
- 反向传播：<br>
<img src="http://latex.codecogs.com/gif.latex?%5Cdelta%20o%5Et%20%3D%20%5Cdelta%20h_i%5Et%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20h_i%5Et%7D%7B%5Cpartial%20o_i%5Et%7D%3D%5Cdelta%20h_i%5Et%5Ctanh%28c_i%5Et%29%20%5C%5C%5B2ex%5D%20%5C%20%5Cdelta%20c%5Et%3D%5Cdelta%20h_i%5Et%5Ccdot%20%5Cfrac%7B%5Cpartial%20h_i%5Et%7D%7B%5Cpartial%20c_i%5Et%7D%3D%5Cdelta%20h_i%5Eto_i%5Et%5B1-%5Ctanh%5E2%28c_i%5Et%29%5D%20%5C%5C%5B2ex%5D%20%5Cdelta%20f%5Et%3D%5Cdelta%20c_i%5Et%20%5Cfrac%7B%5Cpartial%20c_i%5Et%7D%7B%5Cpartial%20f_i%5Et%7D%3D%5Cdelta%20c_i%5Et%5Ccdot%20c_i%5E%7Bt-1%7D%20%5C%5C%5B2ex%5D%20%5Cdelta%20i%5Et%3D%5Cdelta%20c_i%5Et%20%5Cfrac%7B%5Cpartial%20c_i%5Et%7D%7B%5Cpartial%20i%5Et_i%7D%3D%5Cdelta%20c_i%5Et%20%5Chat%20c_i%5Et%20%5C%5C%5B2ex%5D%20%5C%20%5Cdelta%20%5Chat%20c%5Et%3D%5Cdelta%20c_i%5Et%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20c_i%5Et%7D%7B%5Cpartial%20%5Chat%20c_i%5Et%7D%3D%5Cdelta%20c_i%5Et%20%5Ccdot%20i_i%5Et"/><br>
<img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Cfrac%7B%5Cpartial%20E%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D%26%3D%5Csum%20%5Cfrac%7B%5Cpartial%20E%5Et%7D%7B%5Cpartial%20h_i%5Et%7D%5Ccdot%5Cfrac%7B%5Cpartial%20h_i%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D%20%5C%5C%5B2ex%5D%20%26%3D%5Csum%20%5Cdelta%20h_i%5Et%20%5Ccdot%20%5Btanh%28c_i%5Et%29%5D%5Cfrac%7B%5Cpartial%20o_i%5Et%7D%7B%5Cpartial%20h%5E%7Bt-1%7D_i%7D&plus;o_i%5Et%281-%5Ctanh%5E2%28c_i%5Et%29%29%5Ccdot%20%5Cfrac%7B%5Cpartial%20c_i%5Et%7D%7B%5Cpartial%20h_i%5Et%7D%5D%20%5C%5C%5B2ex%5D%20%26%3D%5Csum%20%5B%5Cdelta%20o_i%5Et%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20o_i%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D&plus;%5Cdelta%20c_i%5Et%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20c_i%5Et%7D%7B%5Cpartial%20h%5E%7Bt-1%7D_i%7D%5D%20%5C%5C%5B2ex%5D%20%26%3D%5Csum%20%5C%7B%20%5Cdelta%20o_i%5Et%5Ccdot%20%5Cfrac%7B%5Cpartial%20o_i%5Et%7D%7B%5Cpartial%20h%5E%7Bt-1%7D_i%7D&plus;%5Cdelta%20c_i%5Et%20%5Ccdot%20%5Bc_i%5E%7Bt-1%7D%5Ccdot%20%5Cfrac%7B%5Cpartial%20f_i%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D&plus;%5Chat%20c_i%5Et%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20i%5Et_i%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D&plus;i_i%5Et%5Ccdot%20%5Cfrac%7B%5Cpartial%20%5Chat%20c_i%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D%5D%20%5C%7D%20%5C%5C%5B2ex%5D%20%26%3D%5Csum%28%5Cdelta%20o_i%5Et%5Ccdot%20%5Cfrac%7B%5Cpartial%20o_i%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D&plus;%5Cdelta%20f_i%5Et%5Ccdot%20%5Cfrac%7B%5Cpartial%20f_i%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D&plus;%5Cdelta%20i_i%5Et%5Ccdot%20%5Cfrac%7B%5Cpartial%20i_i%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D&plus;%5Cdelta%20%5Chat%20c_i%5Et%5Ccdot%20%5Cfrac%7B%5Cpartial%20%5Chat%20c_i%5Et%7D%7B%5Cpartial%20h_i%5E%7Bt-1%7D%7D%29%20%5Cend%7Baligned%7D"/><br>
有了各个门的作用可以减缓记忆消减-->是一个累加过程
### Gated Recurrent Unit (GRU)
&nbsp; &nbsp; 相比于LSTM减少到两个门
- 公式：<br>
<img src="http://latex.codecogs.com/gif.latex?z_t%3D%5Csigma%20%28w_z%5Bx_t%3Bh_%7Bt-1%7D%5D&plus;b_z%29%20%5C%5C%5B2ex%5D%20%5C%20r_t%3D%5Csigma%20%28w_r%5Bx_t%3Bj_%7Bt-1%7D%5D&plus;b_r%29%20%5C%5C%5B2ex%5D%20%5Chat%20h_t%3D%5Ctanh%28w_%7B%5Chat%20n%7D%5Bx_t%3Br_n%20%5Ccirc%20h_%7Bt-1%7D%5D&plus;b_%7B%5Chat%20n%7D%29%20%5C%5C%5B2ex%5D%20%5C%20h_t%3D%281-z_t%29%5Ccirc%20h_%7Bt-1%7D%20&plus;%20z_n%5Ccirc%20%5Chat%20h_t"/>
### 负采样
&nbsp; &nbsp; 用在如word2vec中

- [ ] 思想：取一个正样本几个负样本来代替词字典中的所有样本

- [ ] 举例说明： k=4<br>
    
    | contex | word | target |
    | :------: | :------: | :------: |
    | orange | juice | 1 |
    | orange | king | 0 |
    | orange | book | 0 |
    | orange | the | 0 |
    | orange | of | 0 |

    小数据集k=5~20;大数据集k=2~5
- [ ] 挑选负样本方式:<br>
    <img src="http://latex.codecogs.com/gif.latex?p%28w%29%3D%5Cfrac%7B%5Bcounter%28w%29%5D%5E%7B0.75%7D%7D%7B%5Csum_%7Bn%20%5Cin%20D%7D%5Bcounter%28u%29%5D%5E%7B0.75%7D%7D"/><br>
其中counter为w的词频，使用的是平滑策略，为了使低频词出现几率大一些
### Encoder-Decoder
- [ ] 图示：<br>
    <img src="https://caicai.science/images/attention/seq1.png"/>
- [ ] 公式：<br>
     - [ ] Encoder: <img src="http://latex.codecogs.com/gif.latex?h_t%20%3D%20%5Cphi%28h_%7Bt-1%7D%2Cx_t%29%3Df%28W%5E%7B%28hh%29%7Dh_%7Bt-1%7D&plus;W%5E%7B%28hx%29%7Dx_t%20%5C%5C"/>
     - [ ] Decoder: <img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26h_%7BD%2Ct%7D%3D%5Cphi_D%28h_%7Bt-1%7D%2Cc%2Cy_%7Bt-1%7D%29%20%5C%20%5C%20c%3Dh_T%20%5C%5C%5B2ex%5D%20%26%20y_t%3Dsoftmax%28W%28h_t%3Bc%29%29%20%5Cend%7Baligned%7D"/>
### 集束搜索 （Beam Search Algorithm）
- [ ] 公式：<br>
    <img src="http://latex.codecogs.com/gif.latex?arg%20%5Cmax_y%20%5Cfrac%7B1%7D%7BT_y%5E%5Calpha%7D%5Csum_%7Bt%3D1%7D%5E%7BT_y%7D%5Clog%28y%5Et%7Cx%2Cy%5E1%2C...%2Cy%5E%7Bt-1%7D%29"/>其中<img src="http://latex.codecogs.com/gif.latex?%5Calpha%20%5C%20%5C%20usually%20%5C%20be%20%5C%200.7"/>
- [ ] 举例说明：<br>
    假设beam width = 2,预测库里有三个词{a,b,c}<br>

    |  |     预测第一个词 | 预测第二个词 | 预测第三个词 |
    | :----: | :------: | :------: | :------: |
    | 词 |   a,b,c | aa,ab,ac;ba,bb,bc | aba,abb,abc;baa,bab,bac|
    | 概率 | 0.5,0.4,0.1 | 0.15,0.25,0.10;0.2,0.12,0.08| 0.025,0.1,0.125;0.04,0.04,0.12|
    | 处理 | c舍去 | 保留ab,ba|保留abc,bac|
### Bleu score on n-gram
&nbsp; &nbsp; 用于评测机器翻译精度<br>
- [ ] n-gram 概率:<br>
<img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26P_n%20%3D%20%5Cfrac%7B%5Csum_i%20%5Csum_k%20%5Cmin%28h_k%28c_i%29%2C%5Cmax_%7Bj%20%5Cin%20m%7Dh_k%28s_%7Bij%7D%29%29%7D%7B%5Csum_i%5Csum_k%20%5Cmin%28h_k%28c_i%29%29%7D%20%5C%5C%5B2ex%5D%20%26%20%5Ctext%7B%5C%20predict%20%5C%20array%7D%5C%7Bc_1%2Cc_2%2C...c_i%2C...%5C%7D%20%5C%5C%5B2ex%5D%20%26%20%5Ctext%7B%5C%20kth%20%5C%20answer%7D%5C%7Bs_%7B1k%7D%2Cs_%7B2k%7D%2C...%2Cs_%7Bik%7D%2C...%5C%7D%20%5C%5C%5B2ex%5D%20%26%20w_k%20--%3E%5Ctext%7B%5C%20kth%20%5C%20gram%7D%20%5C%5C%5B2ex%5D%20%26%20h_k%28c_i%29%20--%3E%20w_k%5Ctext%7B%5C%20predict%20%5C%20paper%7Dc_i%5Ctext%7B%5C%20count%7D%20%5C%5C%5B2ex%5D%20%26%20h_k%7Bs_%7Bij%7D%7D%20--%3E%20w_k%5Ctext%7B%5C%20the%20%5C%20count%20%5C%20of%20%5C%20jth%20%5C%20answer%7D%20%5Cend%7Baligned%7D"/>
- [ ] 惩罚因子:对译文长度惩罚<br>
<img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%20BP%20%3D%20%5Cbegin%7Bcases%7D1%20%26%20if%20%5C%20l_c%20%3E%20l_s%20%5C%5C%5B2ex%5D%20e%5E%7B1-%5Cfrac%7Bl_s%7D%7Bl_c%7D%7D%20%26%20if%20%5C%20l_c%20%5Cleq%20l_s%20%5Cend%7Bcases%7D%20%5C%5C%5B2ex%5D%20%26%20l_c%20--%3E%20predict%20%5C%20len%20%5C%5C%5B2ex%5D%20%26%20l_s%20--%3E%20answer%20%5C%20len%20%5Cend%7Baligned%7D"/>
- [ ] 最终计算公式:<br>
    <img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%20BLEU%20%3D%20BP%20%5Ctimes%20%5Cexp%28%5Csum_%7Bn%3D1%7D%5ENW_n%5Clog%20P_n%29%20%5C%5C%5B2ex%5D%20%26%20W_n%3D%5Cfrac%7B1%7D%7BN%7D%20%5C%20%5C%20N--%3E%20N%20%5C_Gram--%3EN%20%5Cend%7Baligned%7D"/>
### Attention机制
&nbsp; &nbsp; 生成每个时刻的y，都会利用到x1,x2,x3....，而不再仅仅利用最后时刻的隐藏状态向量。同时注意力机制还能使翻译器使用局部或全局信息
- [ ] BahdanauAttention
  - 图示：<br>
  <img src="https://file.ai100.com.cn/files/sogou-articles/original/071dfd46-8191-413c-a62b-ca93a88a1415/071dfd46-8191-413c-a62b-ca93a88a1415"/>
  - 公式：<br>
    <img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26%20e_%7Bij%7D%3Da%28s_%7Bi-1%7D%2Ch_j%29%20%5C%5C%5B2ex%5D%20%26%20%5Calpha_%7Bij%7D%3D%5Cfrac%7B%5Cexp%28e_%7Bij%7D%29%7D%7B%5Csum_%7Bk%3D1%7D%5E%7BT_x%7D%5Cexp%20%28e_%7Bik%7D%29%7D%20%5C%5C%5B2ex%5D%20%26%20c_i%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BT_x%7D%5Calpha_%7Bij%7Dh_j%20%5Cend%7Baligned%7D"/>
  - 特点：注意力信息由encoder双向RNN的隐藏层与decoder前一时刻的隐藏层贡献
- [ ] LuongAttention
  - 图示：<br>
  <img src="https://ss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=3103722453,2451333191&fm=15&gp=0.jpg"/>
  - 公式：<br>
    <img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%26score%28h_t%2C%5Chat%20h_s%29%20%3D%5Cbegin%7Bcases%7Dh_t%5ET%5Chat%20h_s%20%26%20dot%20%5C%5C%5B2ex%5D%20h_t%5ETW_a%5Chat%20h_s%20%26%20general%20%5C%5C%5B2ex%5D%20v_%7B%5Calpha%7D%5ET%5Ctanh%28W_a%5Bh%5Et%3B%5Chat%20h_s%5D%29%20%26%20concat%20%5Cend%7Bcases%7D%20%5C%5C%5B2ex%5D%20%26%20%5Chat%20h_s%20--%3E%20hidden%20%5C%20of%20%5C%20sth%20%5C%20encoder%20%5C%5C%5B2ex%5D%20%26%20h_t%20--%3E%20hidden%20%5C%20of%20%5C%20tth%20%5C%20decoder%20%5C%5C%5B2ex%5D%20%26%20a_t%28s%29%3Dalign%28h_t%2C%5Chat%20h_s%29%3D%5Cfrac%7B%5Cexp%28score%28h_t%2C%5Chat%20h_s%29%29%7D%7B%5Csum_%7Bs%27%7D%28%5Cexp%28score%28h_t%2C%5Chat%20h_%7Bs%27%7D%29%29%29%7D%20%5Cend%7Baligned%7D"/>
  - 特点：注意力信息由encoder的隐藏层与decoder该时刻的隐藏层贡献
