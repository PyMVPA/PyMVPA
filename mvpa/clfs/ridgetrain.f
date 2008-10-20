C234567
      subroutine ridgetrain(X,y,beta,lm,crit,Nfeatures,Nsamples)
C     Perform ridge regression using gradient descent to save memory

C     This subroutine solves the regression problem
C
C     y = X*beta
C
C     by minimizing the modified error function
C
C     E(beta) = 0.5*(y-X*beta).T*(y-X*beta) + 0.5*lm*beta.T*beta
C
C     via gradient descent. This should use less memory than approaches
C     based on svd or qr-decomposition.

C     There are actually two stopping criteria:
C     1. absolute error goes below a criterion (this criterion is given
C         by the crit argument)
C     2. length of the gradient goes below 0.000001 (this criterion is
C         fixed so far but could in principle become a second element
C         of crit)

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Set things up the right way
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C This is the input
      integer Nfeatures
      integer Nsamples
      real*8 lm
      real*8 X(Nsamples,Nfeatures)
      real*8 y(Nsamples)
      real*8 beta(Nfeatures+1)
      real*8 crit
Cf2py intent(in) X
Cf2py intent(in) y
Cf2py intent(in,out) beta

C This is used internally
      real*8 dE
      real*8 dEtot
      real*8 E

C ymXb is the residual vector
C ymXb could also be recalculated in every step, which would result
C in an additional gain in memory performance. However I can't see
C a way to avoid this without introducing a lot of unneccessary
C overhead.
      real*8 ymXb(Nsamples)

C running indices
      integer sample
      integer feature

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Get y-X*beta and initialize errors
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      E = 0
      dEtot = 1
      do sample = 1,Nsamples
        ymXb(sample) = y(sample) - beta(Nfeatures+1)
        do feature = 1,Nfeatures
          ymXb(sample) = ymXb(sample)
     $                 - X(sample,feature)*beta(feature)
          E = E + ymXb(sample)**2
        end do
      end do

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C     Optimization
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      do while (E .gt. crit .and. dEtot .gt. 0.000001)
        dEtot = 0

C       Update beta ...
        do feature = 1,Nfeatures

C         ... Effect of penalty
          dE = lm*beta(feature)

C         ... Effect of data
          do sample = 1,Nsamples
            dE = dE - ymXb(sample)*X(sample,feature)
          end do

C         ... now we can change beta
          beta(feature) = beta(feature) - 0.01*dE
C         and increment the relative error
          dEtot = dEtot + dE**2
        end do

C       Apply gradient descent to the intercept
        dE = lm*beta(Nfeatures+1)
        do sample = 1,Nsamples
          dE = dE -ymXb(sample)
        end do
        beta(Nfeatures+1) = beta(Nfeatures+1) - 0.01*dE
        dEtot = dEtot + dE**2

C       Determine Error and new ymXb
        E = 0
        do sample = 1,Nsamples
          ymXb(sample) = y(sample) - beta(Nfeatures+1)
          do feature = 1,Nfeatures
            ymXb(sample) = ymXb(sample)
     $                   - X(sample,feature) * beta(feature)
            E = E + ymXb(sample)**2
          end do
        end do
      end do
      return
      end
