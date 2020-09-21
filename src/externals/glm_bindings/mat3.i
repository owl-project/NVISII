// glm::mat3 bindings
// 2018 Dan Wilcox <danomatika@gmail.com>

// ----- detail/type_mat3.hpp -----
%typemap(in) glm::mat3 (void *argp = 0, int res = 0) {
  int res = SWIG_ConvertPtr($input, &argp, $descriptor(glm::mat3*), $disown | 0);
  if (!SWIG_IsOK(res)) 
  { 
    if (!PySequence_Check($input)) {
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " Expected either a sequence or mat3");
      return NULL;
    }

    if (PySequence_Length($input) != 9) {
      PyErr_SetString(PyExc_ValueError,"in method '" "$symname" "', argument " "$argnum" " Size mismatch. Expected 9 elements");
      return NULL;
    }

    float vals[9];
    for (int i = 0; i < 9; i++) {
      PyObject *o = PySequence_GetItem($input,i);
      if (PyNumber_Check(o)) {
        vals[i] = (float) PyFloat_AsDouble(o);
      } else {
        PyErr_SetString(PyExc_ValueError,"in method '" "$symname" "', argument " "$argnum" " Sequence elements must be numbers");      
        return NULL;
      }
    }
    $1 = glm::make_mat3(vals);
  }   
  else {
    glm::mat3 * temp = reinterpret_cast< glm::mat3 * >(argp);
    $1 = *temp;
    if (SWIG_IsNewObj(res)) delete temp;
  }
}

struct mat3 {

    static length_t length();

    mat3();
    mat3(mat3 const & v);
    mat3(float scalar);
    mat3(float x0, float y0, float z0,
         float x1, float y1, float z1,
         float x2, float y2, float z2);
    mat3(vec3 const & v1, vec3 const & v2, vec3 const & v3);
    mat3(glm::mat4 const & m);

    /*mat3 & operator=(mat3 const & m);*/
};

mat3 operator+(mat3 const & m, float scalar);
mat3 operator+(float scalar, mat3 const & m);
mat3 operator+(mat3 const & m1, mat3 const & m2);
mat3 operator-(mat3 const & m, float scalar);
mat3 operator-(float scalar, mat3 const & m);
mat3 operator-(mat3 const & m1, mat3 const & m2);
mat3 operator*(mat3 const & m, float scalar);
mat3 operator*(float scalar, mat3 const & m);
mat3 operator*(mat3 const & m1, mat3 const & m2);
vec3 operator*(vec3 const & v, mat3 const & m);
vec3 operator*(mat3 const & m, vec3 const & v);
mat3 operator/(mat3 const & m, float scalar);
mat3 operator/(float scalar, mat3 const & m);
mat3 operator/(mat3 const & m1, mat3 const & m2);
vec3 operator/(mat3 const & m, vec3 const & v);
vec3 operator/(vec3 const & v, mat3 const & m);
bool operator==(mat3 const & m1, mat3 const & m2);
bool operator!=(mat3 const & m1, mat3 const & m2);

%extend mat3 {

    // [] getter
    // out of bounds throws a string, which causes a Lua error
    vec3 __getitem__(int i) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::mat3::__getitem__()");
            }
            return (*$self)[i-1];
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::mat3::__getitem__()");
            }
            return (*$self)[i];
        #endif
    }

    // [] setter
    // out of bounds throws a string, which causes a Lua error
    void __setitem__(int i, vec3 v) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::mat3::__setitem__()");
            }
            (*$self)[i-1] = v;
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::mat3::__setitem__()");
            }
            (*$self)[i] = v;
        #endif
    }

    // tostring operator
    std::string __tostring() {
        std::stringstream str;
        const glm::length_t width = $self->length();
        const glm::length_t height = (*$self)[0].length();
        for(glm::length_t row = 0; row < height; ++row) {
            for(glm::length_t col = 0; col < width; ++col) {
                str << (*$self)[col][row];
                if(col + 1 != width) {
                    str << "\t";
                }
            }
            if(row + 1 != height) {
                str << "\n";
            }
        }
        return str.str();
    }

    // extend operators, otherwise some languages (lua)
    // won't be able to act on objects directly (ie. v1 + v2)
    mat3 operator+(float scalar) {return (*$self) + scalar;}
    mat3 operator+(mat3 const & m) {return (*$self) + m;}
    mat3 operator-(float scalar) {return (*$self) - scalar;}
    mat3 operator-(mat3 const & m) {return (*$self) - m;}
    mat3 operator*(float scalar) {return (*$self) * scalar;}
    mat3 operator*(mat3 const & m) {return (*$self) * m;}
    vec3 operator*(vec3 const & v) {return (*$self) * v;}
    mat3 operator/(float scalar) {return (*$self) / scalar;}
    mat3 operator/(mat3 const & m) {return (*$self) / m;}
    vec3 operator/(vec3 const & v) {return (*$self) / v;}
    bool operator==(mat3 const & m) {return (*$self) == m;}
    bool operator!=(mat3 const & m) {return (*$self) != m;}
};
