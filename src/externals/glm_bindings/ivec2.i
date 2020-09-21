// glm::ivec2 bindings
// 2018 Dan Wilcox <danomatika@gmail.com>
%typemap(in) glm::ivec2 (void *argp = 0, int res = 0) {
  int res = SWIG_ConvertPtr($input, &argp, $descriptor(glm::ivec2*), $disown | 0);
  if (!SWIG_IsOK(res)) 
  { 
    if (!PySequence_Check($input)) {
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " Expected either a sequence or ivec2");
      return NULL;
    }

    if (PySequence_Length($input) != 2) {
      PyErr_SetString(PyExc_ValueError,"in method '" "$symname" "', argument " "$argnum" " Size mismatch. Expected 2 elements");
      return NULL;
    }

    for (int i = 0; i < 2; i++) {
      PyObject *o = PySequence_GetItem($input,i);
      if (PyNumber_Check(o)) {
        $1[i] = (int) PyLong_AsLong(o);
      } else {
        PyErr_SetString(PyExc_ValueError,"in method '" "$symname" "', argument " "$argnum" " Sequence elements must be numbers");      
        return NULL;
      }
    }
  }   
  else {
    glm::ivec2 * temp = reinterpret_cast< glm::ivec2 * >(argp);
    $1 = *temp;
    if (SWIG_IsNewObj(res)) delete temp;
  }
}

// ----- detail/type_ivec2.hpp -----

struct ivec2 {

    int x, y;

    static length_t length();

    ivec2();
    ivec2(ivec2 const & v);
    ivec2(int scalar);
    ivec2(int s1, int s2);
    ivec2(glm::ivec3 const & v);
    ivec2(glm::ivec4 const & v);

    /*ivec2 & operator=(ivec2 const & v);*/
};

ivec2 operator+(ivec2 const & v, int scalar);
ivec2 operator+(int scalar, ivec2 const & v);
ivec2 operator+(ivec2 const & v1, ivec2 const & v2);
ivec2 operator-(ivec2 const & v, int scalar);
ivec2 operator-(int scalar, ivec2 const & v);
ivec2 operator-(ivec2 const & v1, ivec2 const & v2);
ivec2 operator*(ivec2 const & v, int scalar);
ivec2 operator*(int scalar, ivec2 const & v);
ivec2 operator*(ivec2 const & v1, ivec2 const & v2);
ivec2 operator/(ivec2 const & v, int scalar);
ivec2 operator/(int scalar, ivec2 const & v);
ivec2 operator/(ivec2 const & v1, ivec2 const & v2);
/*ivec2 operator%(ivec2 const & v, int scalar);
ivec2 operator%(int scalar, ivec2 const & v);
ivec2 operator%(ivec2 const & v1, ivec2 const & v2);*/
bool operator==(ivec2 const & v1, ivec2 const & v2);
bool operator!=(ivec2 const & v1, ivec2 const & v2);

%extend ivec2 {
    
    // [] getter
    // out of bounds throws a string, which causes a Lua error
    int __getitem__(int i) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::ivec2::__getitem__()");
            }
            return (*$self)[i-1];
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::ivec2::__getitem__()");
            }
            return (*$self)[i];
        #endif
    }

    // [] setter
    // out of bounds throws a string, which causes a Lua error
    void __setitem__(int i, int f) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::ivec2::__setitem__()");
            }
            (*$self)[i-1] = f;
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::ivec2::__setitem__()");
            }
            (*$self)[i] = f;
        #endif
    }

    // tostring operator
    std::string __tostring() {
        std::stringstream str;
        for(glm::length_t i = 0; i < $self->length(); ++i) {
            str << (*$self)[i];
            if(i + 1 != $self->length()) {
                str << " ";
            }
        }
        return str.str();
    }

    // extend operators, otherwise some languages (lua)
    // won't be able to act on objects directly (ie. v1 + v2)
    ivec2 operator+(ivec2 const & v) {return (*$self) + v;}
    ivec2 operator+(int scalar) {return (*$self) + scalar;}
    ivec2 operator-(ivec2 const & v) {return (*$self) - v;}
    ivec2 operator-(int scalar) {return (*$self) - scalar;}
    ivec2 operator*(ivec2 const & v) {return (*$self) * v;}
    ivec2 operator*(int scalar) {return (*$self) * scalar;}
    ivec2 operator/(ivec2 const & v) {return (*$self) / v;}
    ivec2 operator/(int scalar) {return (*$self) / scalar;}
    /*ivec2 operator%(ivec2 const & v) {return (*$self) % v;}
    ivec2 operator%(int scalar) {return glm::ivec2((int)((*$self)[0]) % (int)scalar,(int)((*$self)[1]) % (int)scalar);}*/
    bool operator==(ivec2 const & v) {return (*$self) == v;}
    bool operator!=(ivec2 const & v) {return (*$self) != v;}
};
