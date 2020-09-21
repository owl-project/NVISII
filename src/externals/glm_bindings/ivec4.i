// glm::ivec4 bindings

// ----- detail/type_ivec4.hpp -----
%typemap(in) glm::ivec4 (void *argp = 0, int res = 0) {
  int res = SWIG_ConvertPtr($input, &argp, $descriptor(glm::ivec4*), $disown | 0);
  if (!SWIG_IsOK(res)) 
  { 
    if (!PySequence_Check($input)) {
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " Expected either a sequence or ivec4");
      return NULL;
    }

    if (PySequence_Length($input) != 4) {
      PyErr_SetString(PyExc_ValueError,"in method '" "$symname" "', argument " "$argnum" " Size mismatch. Expected 4 elements");
      return NULL;
    }

    for (int i = 0; i < 4; i++) {
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
    glm::ivec4 * temp = reinterpret_cast< glm::ivec4 * >(argp);
    $1 = *temp;
    if (SWIG_IsNewObj(res)) delete temp;
  }
}

struct ivec4 {
    
    int x, y, z, w;

    static length_t length();

    ivec4();
    ivec4(ivec4 const & v);
    ivec4(int scalar);
    ivec4(int s1, int s2, int s3, int s4);
    ivec4(ivec2 const & a, ivec2 const & b);
    ivec4(ivec2 const & a, int b, int c);
    ivec4(int a, ivec2 const & b, int c);
    ivec4(int a, int b, ivec2 const & c);
    ivec4(ivec3 const & a, int b);
    ivec4(int a, ivec3 const & b);

    /*ivec4 & operator=(ivec4 const & v);*/
};

ivec4 operator+(ivec4 const & v, int scalar);
ivec4 operator+(int scalar, ivec4 const & v);
ivec4 operator+(ivec4 const & v1, ivec4 const & v2);
ivec4 operator-(ivec4 const & v, int scalar);
ivec4 operator-(int scalar, ivec4 const & v);
ivec4 operator-(ivec4 const & v1, ivec4 const & v2);
ivec4 operator*(ivec4 const & v, int scalar);
ivec4 operator*(int scalar, ivec4 const & v);
ivec4 operator*(ivec4 const & v1, ivec4 const & v2);
ivec4 operator/(ivec4 const & v, int scalar);
ivec4 operator/(int scalar, ivec4 const & v);
ivec4 operator/(ivec4 const & v1, ivec4 const & v2);
/*ivec4 operator%(ivec4 const & v, int scalar);
ivec4 operator%(int scalar, ivec4 const & v);
ivec4 operator%(ivec4 const & v1, ivec4 const & v2);*/
bool operator==(ivec4 const & v1, ivec4 const & v2);
bool operator!=(ivec4 const & v1, ivec4 const & v2);

%extend ivec4 {

    // [] getter
    // out of bounds throws a string, which causes a Lua error
    int __getitem__(int i) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::ivec4::__getitem__()");
            }
            return (*$self)[i-1];
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::ivec4::__getitem__()");
            }
            return (*$self)[i];
        #endif
    }

    // [] setter
    // out of bounds throws a string, which causes a Lua error
    void __setitem__(int i, int f) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::ivec4::__setitem__()");
            }
            (*$self)[i-1] = f;
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::ivec4::__setitem__()");
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
    ivec4 operator+(ivec4 const & v) {return (*$self) + v;}
    ivec4 operator+(int scalar) {return (*$self) + scalar;}
    ivec4 operator-(ivec4 const & v) {return (*$self) - v;}
    ivec4 operator-(int scalar) {return (*$self) - scalar;}
    ivec4 operator*(ivec4 const & v) {return (*$self) * v;}
    ivec4 operator*(int scalar) {return (*$self) * scalar;}
    ivec4 operator/(ivec4 const & v) {return (*$self) / v;}
    ivec4 operator/(int scalar) {return (*$self) / scalar;}
    /*ivec4 operator%(ivec4 const & v) {return (*$self) % v;}
    ivec4 operator%(int scalar) {return (*$self) % scalar;}*/
    bool operator==(ivec4 const & v) {return (*$self) == v;}
    bool operator!=(ivec4 const & v) {return (*$self) != v;}
};
