# **Tutorial 2 Further Reading**

## **IAM Model Summary** 
The economy model in most IAMs is a capital accumulation model. 
- Capital combines with a laboring population and technology to generate productivity ($Y$) that is hindered by climate damage. 
- A savings fraction of production drives capital accumulation, while the rest is consumed. Welfare is determined by consumption.
- Climate action is formulated by a mitigation rate, $\mu$, which along with the savings rate are the two control parameters in the model.
- These are used to maximize welfare.

The climate model in DICE could be improved (c.f. [this study](https://www3.nd.edu/~nmark/Climate/DICE-simplified_2019.pdf)). We only summarize here how it interacts with the economy model:
- Productivity generates industrial emissions, $$E_\mathrm{ind}=(1-\mu)\sigma Y,$$ where the $1-\mu$ factor accounts for reduced carbon intensity of production, $\sigma$, via supply-side mitigation measures (e.g. increased efficiency). 
- The productivity $Y$ rather than output production ($Q$) see model is used here because damages aren't included. 
- Namely, the emissions produced in the process of capital production occur before climate change has a chance to inflict damage on the produced output. 
- These emissions combine with natural emissions to drive the temperature changes appearing in the damage function, closing the economy-climate loop. 

Here are a list of variables used:
- $K$ capital
- $Y$ productivity
- $Q$ production
- $A$ technology conversion
- $S$ savings rate
- $\mu$ mitigation rate
- $\Lambda$ mitigation cost
- $\Omega$ damage fraction of productivity
- $\sigma$ carbon intensity of production
- $E$ emissions

## **Exogeneous Control Variables**
There are two exogeneous variables in the model:
- mitigation (emissions reduction) rate $\mu_t$, and 
- the savings rate $S_t$. 

There exists a mitigation cost associated with a specific mitigation rate, presented as a fraction of productivity, $\Lambda_t=\theta_1\mu^{\theta_2}$. This implies that increased mitigation efforts correspond to elevated costs.

The savings rate $S_t$ extracts a portion of the total production to invest, thereby boosting capital. The rest of the production is then allocated for consumption.

## **Economy Model Summary**
The essence of the economy model in DICE is a capital accumulation model. Existing capital depreciates at rate $\delta$ and new capital arises through investment,
$$K_{t+1}=(1-\delta)K_t+I_t$$
where the invested capital $I=SQ$ is determined by a chosen fraction $S$ of the production $Q$ that is "saved" rather than consumed. Production is given as the productivity $Y$ reduced by damages and mitigation cost, $$Q=(1-\Omega)(1-\Lambda)Y.$$ Productivity, $$Y=A K^\gamma L^{1-\gamma},$$
is determined by the technology conversion $A$ operating on a combination of capital $K$ and labor $L$ whose relative contributions are set by the capital elasticity parameter $\gamma$. Labor is population which set to saturate over the 2nd half of the 21st century. Technology conversion is only weakly sigmoidal in time, deviating slightly from linear growth.

The remaining production is consumed $$C:=(1-S)Q$$ producing utility $$U(C,L)=Lu(c)=Lu(C/L),$$ using the isoelastic utility function, $$u(c)=\frac{(c+1)^{1-\alpha}-1}{1-\alpha}.$$ The overall value of a projected future is then $$V=\sum_{t=1}^{\infty}\gamma^t U(C_t,L_t),$$ where $\gamma=1/(1+\rho)$ for discount rate $\rho$ and we use the population level utility function $U(C,L)=Lu(C/L)$.


Here are a list of variables used:
- $K$ capital
- $Y$ productivity
- $Q$ production
- $A$ technology conversion
- $S$ savings rate
- $\mu$ mitigation rate
- $\Lambda$ mitigation cost
- $\Omega$ damage fraction of productivity
- $\sigma$ carbon intensity of production
- $E$ emissions

## **Optimal Planning**
The unconstrained problem aims to maximize $V$ within the bounds of $\mu_t$ and $S_t$ time courses (while considering constraints on $\mu$ and $S$). Why is there a "sweet spot"? Increasing savings boosts investment and productivity, but higher production leads to emissions, resulting in increased temperature and damages that reduce production. Mitigation costs counterbalance this effect, creating a trade-off. As a result, there typically exists a meaningful joint time series of $\mu_t$ and $S_t$ that maximizes $V$. Due to the discount factor $\gamma$, $V$ depends on the future consumption sequence (non-invested production) within a few multiples of the horizon, approximately $1/(1-\gamma)$ time steps into the future.

Constrained formulations introduce an additional constraint to limit industrial emissions below a certain threshold.

## **Social Cost of Carbon**
A definition for the social cost of carbon (SCC) is 

- *the decrease in aggregate consumption in that year that would change the current...value of social welfare by the same amount as a one unit increase in carbon emissions in that year.* ([Newbold, Griffiths, Moore, Wolverton, & Kopits, 2013](https://www.worldscientific.com/doi/abs/10.1142/S2010007813500012)). 

The $SCC$ quantifies how much consumption is lost with increased emissions, using changes in welfare to make the connection. In technical terms: 

- the marginal value with respect to emissions relative to the marginal value with respect to consumption, $$SCC_t\propto\frac{\partial V/\partial E_t}{\partial V/\partial C_t}=\frac{\partial C_t}{\partial E_t}.$$
This is usually expressed by multiplying by a proportionality factor of $-1000$ that converts the units to 2010 US dollars per tonne of CO2. 

# **Tutorial 4 Further Reading**

## **Vectorization Methods for Creating Word Clouds**

Let's write down what they compute by denoting the index, $d$, over the $D$ documents and the index, $w$, over the $W$ words in the vocabulary (the list of all the words found in all the tweets, which we'll call documents): 
- term frequency, $\mathrm{tf}(w,d)$. The frequency of a word $w$ in a document $d$ is $$\mathrm{tf}(w,d):=\frac{n(w,d)}{n(d)},$$ where $n(w,d)$ is the number of times term $w$ is in document $d$ and $n(d)=\sum_{w=1}^W n(w,d)$ is the total number of words in document $d$. The term frequency over all the documents is then, $$\mathrm{tf}(w):=\frac{\sum_{d=1}^D n(d)\mathrm{tf}(w,d)}{N},$$ where the denominator $N=\sum_{d=1}^D n(d)$ is just the total word count across all documents.
- term frequency-inverse document frequency, $\mathrm{Tfidf}(w,d):=\mathrm{tf}(w,d)\mathrm{idf}(w)$. Here, $$\mathrm{idf}(w)=\frac{\log(D+1)}{\log(n(w)+1)+1},$$ where $n(w)$ is the number of documents in which term $t$ appears, i.e. $n(w,d)>0$. Idf is like an inverse document frequency. The `sklearn` package then uses $$\mathrm{Tfidf}(w)=\frac{1}{D}\sum_{d=1}^D \frac{\mathrm{Tfidf}(w,d)}{||\mathrm{Tfidf}(\cdot,d)||},$$ where $||\vec{x}||=\sqrt{\sum_{i=1}^Nx^2_i}$ is the Euclidean norm.

$\mathrm{Tfidf}$ aims to add more discriminability to frequency as a word relevance metric by downweighting words that appear in many documents since these common words are less discriminative.