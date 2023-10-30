#define GET_WB(a)  Matrix<T> W; Vector<T> b; get_wb(a,W,b); 
#define GET_w(a)  Vector<T> w; get_w(a,w); 

class Data:

    def __init__(X : torch.Tensor, intercept : bool):
        self.x = X
        self.scale_intercept = 1.0
        self.intercept = intercept
        self.norms = None

    @abc.abstractmethod
    def pred(input : torch.Tensor, output : torch.Tensor) -> None: 
        return 

    @abc.abstractmethod
    def add_dual_pred(input : torch.Tensor, output : torch.Tensor, a -> float : 1.0, b -> float : 1.0) -> None: 
        return  

    @abc.abstractmethod
    def print() -> None: 
        return 

    def get_coordinates(const int ind, Vector<typename M::index_type>& indices) const {
        if (_X.is_sparse) {
            typename M::col_type col;
            _X.refCol(ind, col);
            col.refIndices(indices);
        }
    };

    @abc.abstractmethod
    def pred(const int ind, const D& input, typename D::element& output) -> None: 
        return 

    @abc.abstractmethod
    def is_sparse() -> bool:
        return self.x.is_sparse

    @abc.abstractmethod
    def set_intercept(const D& x0, D& x) -> None: 
        return 

    @abc.abstractmethod
    def reverse_intercept(D& x) -> None: 
        return 

    def intercept() -> bool:
        return self.intercept


class DataLinear(Data):

    DataLinear(const M& X, const bool
        intercept = false) : Data<M, D>(X, intercept) { };

    
    virtual void print() const {
        logging(logINFO) << "Matrix X, n=" << _X.n() << ", p=" << _X.m();
    };
    

private:
    inline void get_w(const Vector<T>& input, Vector<T>& w) const {
        const int n = input.n();
        input.refSubVec(0, n - 1, w);
    };
}

"""
/// prediction = W*X
template <typename M>
class DataMatrixLinear final : public Data<M, Matrix<typename M::value_type> > {
    typedef M data_type;
    typedef typename M::value_type T;
    typedef Matrix<T> D;
    using Data<M, D>::_X;
    using Data<M, D>::_scale_intercept;
    using Data<M, D>::_intercept;

public:
    typedef Matrix<T> variable_type;
    DataMatrixLinear(const M& X, const bool
        intercept = false) : Data<M, D>(X, intercept) {
        _ones.resize(_X.n());
        _ones.set(T(1.0));
    };

    // _X  is  p x n
    // input is nclass x p
    // output is nclass x n
    inline void pred(const Matrix<T>& input, Matrix<T>& output) const {
        if (_intercept) {
            GET_WB(input);
            W.mult(_X, output);
            output.rank1Update(b, _ones);
        }
        else {
            input.mult(_X, output);
        }
    };
    inline void pred(const int ind, const Matrix<T>& input, Vector<T>& output) const {
        typename M::col_type col;
        _X.refCol(ind, col);
        if (_intercept) {
            GET_WB(input);
            W.mult(col, output);
            output.add(b, _scale_intercept);
        }
        else {
            input.mult(col, output);
        }
    };
    inline void add_dual_pred(const Matrix<T>& input, Matrix<T>& output, const T a1 = T(1.0), const T a2 = T(1.0)) const {
        if (_intercept) {
            output.resize(input.m(), _X.m() + 1);
            GET_WB(output);
            input.mult(_X, W, false, true, a1, a2); //  W = input * X.T =  (X* input.T).T
            input.mult(_ones, b, a1, a2);
        }
        else {
            input.mult(_X, output, false, true, a1, a2);
        }
    };
    inline void add_dual_pred(const int ind, const Vector<T>& input, Matrix<T>& output, const T a = T(1.0), const T bb = T(1.0)) const {
        typename M::col_type col;
        _X.refCol(ind, col);
        if (bb != T(1.0)) output.scal(bb);
        if (_intercept) {
            output.resize(input.n(), _X.m() + 1);
            GET_WB(output);
            W.rank1Update(input, col, a);
            b.add(input, a * _scale_intercept);
        }
        else {
            output.rank1Update(input, col, a);
        }
    };
    virtual void print() const {
        logging(logINFO) << "Matrix X, n=" << _X.n() << ", p=" << _X.m();
    };
    virtual DataLinear<M>* toDataLinear() const {
        return new DataLinear<M>(_X, _intercept);
    };
    virtual void reverse_intercept(Matrix<T>& x) {
        const int m = x.m();
        const int n = x.n();
        if (_scale_intercept != T(1.0))
            for (int ii = 0; ii < n; ++ii)
                x[ii * m + m - 1] *= _scale_intercept;
    };
    virtual void set_intercept(const Matrix<T>& x0, Matrix<T>& x) {
        _scale_intercept = sqrt(T(0.1) * _X.normFsq() / _X.n());
        _ones.set(_scale_intercept);
        x.copy(x0);
        const int m = x.m();
        const int n = x.n();
        for (int ii = 0; ii < n; ++ii)
            x[ii * m + m - 1] /= _scale_intercept;
    };



private:
    inline void get_wb(const Matrix<T>& input, Matrix<T>& W, Vector<T>& b) const {
        const int p = input.n();
        input.refSubMat(0, p - 1, W);
        input.refCol(p - 1, b);
    };
    Vector<T> _ones;
};
"""
