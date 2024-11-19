# ImportanceSampling
The Jupyter Notebook can be found here: https://colab.research.google.com/drive/15BVtn7loJOJv4nUY5KpcT6yYescXNmQO?usp=sharing
Some values may differ from those reported in this README file because the script was ran again.
## 1
In this question, we generate data from a logistic regression model, the "logistic function" gets a 2d data point and returns a
probability p given by the logistic model formula. Then, we use this probability to generate 0 or 1 using np.random.binomial.
0 or 1s are the ys. Based on the size of the sample we generate np arrays of size 2*10,50,100 that store covariates from uniform[-2,2] and then use them to generate their corresponding binary values which are distributed according to the logistic regression model. These different np arrays are X10,X50,X100 and y10,y50,y100. The number of 0s and 1s are printed as instructed by the question. 
## 2
### functions
I begin by elaborating on the functions. To do so, it is important to explain the task in detail. What we are trying to achieve here is approximating the underlying beta given the data we generated. Intuitively We use importance sampling to sample from the joint distribution of data and beta. To do this, we have a gaussian proposal and sample from it 20000 times. Each time w=p(data|beta)p(beta)/q(beta), where q is the proposal and p(beta) as a prior has a gaussian distribution. p(data|b) is also known as the likelihood and is the product of (yi|x1,x2,beta). Now that we have the foundation, we use log likelihood and log weights etc to avoid numerical issues. Obviously, products will become sums under this setting. Lastly, the expectation of betas we sampled using this technique will be close to the initial betas.

def log likelidhood(b,X,y): returns the log-likelihood of observing the data given beta

proposal_betas: samples from the proposal distribution (I chose std=1.5)

def beta_approx(X,y): computes weights given the log of the w=p(data|beta)p(beta)/q(beta) equation.

 the weights are normalized by subtracting the maximum and dividing them by their sum.

 The means (approximated betas) are calculated using the weights and the proposed betas.

 weights and approximations are returned.

 def compute_error_estimates:

 This function gets proposal values, weights and posterior means (approximated) and returns the standard error.
 
### Applied to data
We then proceed to apply the said functions to the generated samples of size 10,50,100

For sample size 10 the posterior means are: [ 0.91635967  0.66523389 -0.40596743]

with their standard errors being: [0.6592511950670328, 0.6855253121238374, 0.502932610464077]

and the effective sample size being: 2837.769177320035

These values are computer for sample sizes 50 and 100 in the same fashion.


It is worth noting that the std decreased with increase in sample size as the number of data points change the likelihood term in weight calculation, However the effective sample size decreased. This is not always the case and in this question it is mostly due to the fact that gaussian proposal is not adapt for the target distribution, with increase in data points, the likelihood function becomes more concentrated causes the proposal to fail to cover the now narrower distribution.

Lastly, I resample 10000 betas using the normalized weights, this means choosing 10000 betas from 20000 at random with the distribution dictated by the normalize weights (normalizing the weights allowed them to have sum=1, each corresponding to a probability. The histograms show that through the importance sampling process, the proposal distribution which was initially gaussian with mean 0 and cov 1.5(I) slowly shifted towards a mean that is close to the sought after betas. 

### Standard Optimizer
Lastly I used scipy.minimize using the BFGS algorithm as an standard optimizer. To maximize likelihood, I aim to minimize negative log-likelihood.

The arguments for this function are the function we aim to minimize, the data points, an initial vector (I used zero) and the method (I chose BFGS). I used sample of size 100 to make the comparison and the posterior means I got using the importance sampling method were almost as close to the real values as the ones the standard optimizer computed.
## 3
In this section, 6 new betas and covariates were added. I made some changes to the code used in question 1 and 2 to make the code more reusable such as def make_data(n) which generates a sample of size n. Other than that the procedures stay the same.
After repeating all of the experiment, it is obvious that the effective sample size has drastically decreased. Especially in n=50 and n=100, the ess is 4 and 1.6 respectively which renders the importance sampling almost useless. When dimension increases, the ess formula becomes highly sensitive to few dominating weights.

## 4
### functions:
log_posterior(beta, X, y), This function returns the function we want to minimize during this whole assignment, negative log of posterior likelihood, however we use it to find and optimal mode using standard optimizers (rather than mean which is our task).

mode_finder(X,y): This uses scipy BFGS optimizer to find an optimal mode given our data points. I used a zero vector as The initial beta vector for the minimize function's argument. 

approx(X,y,prop_center,prop_cov): This is the same as the other functions we used to implement importance sampling, the difference is that it has a prop_center and cov argument which are theoretically the mean and cov matrix of the proposal gaussian. In this question we only change the mean to the optimal mod we found for each case.

### Applied to data:
We apply the new technique to samples of size 10, 50 and 100

In all cases the ess increased. The new ess for 10,50,100 were 1230, 10.8, 6.4 respectively. Which is an improvement from centering the proposal at 0
## 5
To address the dependency of parameters on posterior. A technique suggested in section 9.7 of this article: https://artowen.su.domains/mc/Ch-var-is.pdf will be used.

Namely, the inverse Hessian of the negative log-posterior at mode is used. This is very helpful since mode has already been computed using optimizers and now computing the inverse Hessian at mode is less computationally costly. I implemented this method for n=100 which had the least ess to observe the difference. The optimizer method I used in this question is different because BFGS caused some data type issues I was unable to resolve. The centering procedure is the same as part 4, however instead of having a scalar*I covariance matrix, the covariance will be Hessian of the negative log posterior at mode.

By using this method, I was able to increase ess for n=100 to over 60 and for n=50 to over 400


