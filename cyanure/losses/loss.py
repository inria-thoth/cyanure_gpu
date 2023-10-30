import abc
import math
import torch

from typing import Tuple

class Loss:
    def __init__(self, data : torch.Tensor, y : torch.Tensor, intercept : bool) -> None:
        self.intercept = intercept
        self.scale_intercept = 1.0
        self.input_data = data
        self.labels = y
        self.norms_vector = None

    # functions that should be implemented in derived classes (see also in protected section)
    @abc.abstractmethod
    def eval_tensor(self, input : torch.Tensor) -> float:
        return

    @abc.abstractmethod
    def eval(self, input : torch.Tensor, i : int) -> float:
        return

    # should be safe if output=input1
    @abc.abstractmethod
    def add_grad(self, input : torch.Tensor, i : int, output : torch.Tensor, eta : float = 1.0) -> None:
        return 

    @abc.abstractmethod
    def scal_grad(self, input : torch.Tensor, i : int) -> float:
        return

    @abc.abstractmethod
    def add_feature(self, input : torch.Tensor, output : torch.Tensor, s : float)-> None:
        return

    @abc.abstractmethod
    def add_feature(self, output : torch.Tensor, i : int, s : float) -> None:
        return

    @abc.abstractmethod
    def fenchel(self, input : torch.Tensor) -> float:
        return

    @abc.abstractmethod
    def print(self) -> None:
        return 

    # virtual functions that may be reimplemented, if needed
    def double_add_grad(self, input1 : torch.Tensor, input2 : torch.Tensor, i : int, output : torch.Tensor, eta1 : float = 1.0, eta2 : float = -1.0, dummy : float = 1.0) -> None:
        add_grad(input1,i,output,eta1)
        add_grad(input2,i,output,eta2)

    def provides_fenchel(self) -> bool:
        return True

    def norms_data(self, norms : torch.Tensor) -> torch.Tensor:
        if (self.norms_vector is None or self.norms_tensor.size(dim=0) == 0):

            self.norms_tensor = torch.Tensor((self.input_data.size(dim=0)))

            for i in range(self.input_data.size(dim=0)):
                self.norms_tensor[i] = torch.pow(torch.linalg.vector_norm(self.input_data[:, i]), 2)

            if (self.intercept):
                norms = norms + math.pow(self.scale_intercept)
        
        return torch.clone(self.norms_tensor)

    def norms(self, ind : int) -> float:
        return self.norms[:, ind]

    def lipschitz(self) -> float:
        norms = self.norms_data(norms)
        return self.lipschitz_constant()*torch.max(norms)

    def lipschitz_li(self, Li : torch.Tensor) -> float:
        Li = self.norms_data(Li)
        Li = Li * self.lipschitz_constant()
        return Li
    
    def transpose(self) -> bool:
        return False

    def grad(self,input : torch.Tensor) -> torch.Tensor:
        tmp = self.get_grad_aux(input)
        return self.add_dual_pred_tensor(tmp,1.0/tmp.size(dim=0));
    
    def eval_random_minibatch(self, input: torch.Tensor, minibatch : int) -> float:
        sum = 0
        n = self.n()
        for ii in range(minibatch): 
            sum += self.eval(input,random.randint(0, sys.maxsize) % n)
        return (sum/minibatch)
    
    def grad_random_minibatch(self, input : torch.Tensor, grad : torch.Tensor, minibatch : int) -> torch.Tensor:
        n = self.n()
        for ii in range(minibatch):
            coef = 0.0 if ii == 0 else 1.0
            self.add_grad(input, random.randint(0, sys.maxsize) % n, grad, coef)       
        return grad * (1.0/minibatch)
    
    def kappa(self) -> float:
        return 0

    def set_anchor_point(self, z : torch.Tensor) -> None:
        pass

    def get_anchor_point(self, z : torch.Tensor) -> None:
        pass

    def get_dual_variable(self, input : torch.Tensor, grad2 : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad1 = self.get_grad_aux(input)
        grad1 = self.get_dual_constraints(grad1)
        return grad1, self.add_dual_pred_tensor(grad1,1.0/grad1.size(dim=0))
    
    # non-virtual function classes
    def n(self) -> int:
        return self.labels.size(dim=1)

    def get_coordinates(ind : int, indices : torch.Tensor) -> None: 
        if (self.input_data.is_sparse):
            col = self.input_data.refCol[: , ind]
            indices = col
            
    def set_intercept(self, initial_weight : torch.Tensor, weight: torch.Tensor) -> None:
        self.scale_intercept = math.sqrt(0.1 * torch.pow(torch.linalg.vector_norm(self.input_data), 2) / self.input_data.size(dim=1));
        weight = torch.clone(initial_weight)
        weight[weight.size(dim=1) - 1] /= self.scale_intercept

    def reverse_intercept(self, weight: torch.Tensor) -> None:
        if (self.scale_intercept != 1.0):
            weight[weight.size(dim=1) - 1] *= self.scale_intercept

    def check_grad(self, input : torch.Tensor, output : torch.Tensor) -> None:
        output = torch.clone(input)
        x1 = torch.clone(input)
        x2 = torch.clone(input)
        for i in range(input.size(dim=1)):
            x1[ii] -= 1e-7
            x2[ii] += 1e-7
            output[ii]=(self.eval_tensor(x2)-self.eval_tensor(x1))/(2e-7);
            x1[ii] += 1e-7
            x2[ii] -= 1e-7

    @abc.abstractmethod
    def get_grad_aux(self, input : torch.Tensor)-> None:
        return

    @abc.abstractmethod
    def lipschitz_constant(self) -> float:
        return

    @abc.abstractmethod
    def get_dual_constraints(self, grad1 : torch.Tensor) -> torch.Tensor:
        return 

    def add_dual_pred_tensor(self, input : torch.Tensor, a : float = 1.0) -> torch.Tensor:
        weight = None
        if (self.intercept):
            m = self.input_data.size(dim=0)
            output = torch.Tensor((1, m + 1))
            weight = get_w(input, weight)
            weight = a * torch.matmul(self.input_data, input) + weight * b
            output[:, m] = self.scale_intercept * a * input.sum()
        else:
            output = a * torch.matmul(self.input_data, input)

        return output

    def add_dual_pred(self, ind : int, output : torch.Tensor, a : float = 1.0, b : float = 1.0) -> torch.Tensor:
        col = self.input_data[ind, :]
        if (self.intercept):
            m = self.input_data.size(dim=0)
            output.resize_((1, m + 1))
            weight = get_w(input, weight)
            col = a * weight +  b * col 
            output[:, m] = a * _scale_intercept + b * output[:, m]
        else:
            output.resize_((1, m))
            output = a * weight +  b * output 

        return output
        
    def pred_tensor(self, input : torch.Tensor) -> torch.Tensor:
        weight = None
        if (self.intercept): 
            weight = get_w(input, weight)
            output = torch.matmul(torch.transpose(self.input_data, 0, 1), weight)
            return output  + (input[input.size(dim=1) - 1] * self.scale_intercept)
        else:
            return torch.matmul(torch.transpose(self.input_data, 0, 1), input)

    def pred(ind : int, input: torch.Tensor) -> float:
        col = None
        weight = None
        col = self.input_data[ind, :]
        if (self.intercept):
            get_w(input, weight)
            return torch.dot(col, weight) + input[:, input.n() - 1] * self.scale_intercept
        else:
            return torch.dot(col, input)

    def get_w(input : torch.Tensor, weight : torch.Tensor) -> torch.Tensor:
        n = input.n()
        if (len(weight.size()) == 1):
            weight = input[:n-1]
        else:
            weight = input[:, :n-1]

        return weight

"""
class ProximalPointLoss:

      ProximalPointLoss(loss : Loss, const D& z, const T kappa) 
         : loss_type(loss.data(), loss.y()), _loss(loss), _kappa(kappa)  { 
            _z.copy(z);
            this->_id=PPA;
         };

      inline T eval(const D& input) const {
         D tmp;
         tmp.copy(input);
         tmp.sub(_z);
         return _loss.eval(input)+T(0.5)*_kappa*tmp.nrm2sq();
      };
      inline T eval(const D& input, const INTM i) const {
         D tmp;
         tmp.copy(input);
         tmp.sub(_z);
         return _loss.eval(input,i)+T(0.5)*_kappa*tmp.nrm2sq();
      };
      inline void grad(const D& input, D& output) const {
         _loss.grad(input,output);
         output.add(input,_kappa);
         output.add(_z,-_kappa);
      };
      inline void add_grad(const D& input, const INTM i, D& output, const T eta = T(1.0)) const {
         _loss.add_grad(input,i,output,eta);
         output.add(input,_kappa*eta);
         output.add(_z,-_kappa*eta);
      };
      inline void double_add_grad(const D& input1, const D& input2, const INTM i, D& output, const T eta1 = T(1.0), const T eta2 = -T(1.0), const T dummy = T(1.0)) const {
         _loss.double_add_grad(input1,input2,i,output,eta1,eta2);
         if (dummy) {
            output.add(input1,dummy*_kappa*eta1);
            output.add(input2,dummy*_kappa*eta2);
            if (abs<T>(eta1+eta2) > EPSILON)
               output.add(_z,-_kappa*dummy*(eta1+eta2));
         }
      }
      virtual T eval_random_minibatch(const D& input, const INTM minibatch) const {
         const T sum=_loss.eval_random_minibatch(input,minibatch);
         D tmp;
         tmp.copy(input);
         tmp.sub(_z);
         return sum+T(0.5)*_kappa*tmp.nrm2sq();
      };
      virtual void grad_random_minibatch(const D& input, D& grad, const INTM minibatch) const {
         _loss.grad_random_minibatch(input,grad,minibatch);
         grad.add(input,_kappa);
         grad.add(_z,-_kappa);
      };
      inline void print() const {
         logging(logINFO) << "Proximal point loss with";
         _loss.print();
      }
      virtual bool provides_fenchel() const { return false; };
      virtual T fenchel(const D& input) const { return 0; };
      virtual T lipschitz() const { 
         return _loss.lipschitz() + _kappa;
      };
      virtual void lipschitz(Vector<T>& Li) const {
         _loss.lipschitz(Li);
         Li.add(_kappa);
      };
      virtual void scal_grad(const D& input, const INTM i, typename D::element& output) const  { 
         _loss.scal_grad(input,i,output);
      };
      virtual void  add_feature(const D& input, D& output, const T s) const { 
         _loss.add_feature(input,output,s);
      };
      virtual void  add_feature(D& output, const INTM i, const typename D::element& s) const {
         _loss.add_feature(output,i,s);
      };
      virtual void set_anchor_point(const D& z) { _z.copy(z); };
      virtual void get_anchor_point(D& z) const { z.copy(_z); };

      def kappa(): 
        return _kappa

      def transpose() -> bool: 
         return self.loss.transpose()
   
    @abc.abstractmethod
    def get_grad_aux(input : torch.Tensor, grad1 : torch.Tensor) -> None:
        return

    @abc.abstractmethod
    def lipschitz_constant() -> int:
         return

    @abc.abstractmethod
    def get_dual_constraints(grad1 : torch.Tensor):
        return

   
   private:
      const loss_type& _loss;
      const T _kappa;
      D _z;
"""
