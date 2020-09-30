// glm::ivec3 bindings
%typemap(in) glm::ivec3 (void *argp = 0, int res = 0) {
  int res = SWIG_ConvertPtr($input, &argp, $descriptor(glm::ivec3*), $disown | 0);
  if (!SWIG_IsOK(res)) 
  { 
    if (!PySequence_Check($input)) {
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " Expected either a sequence or ivec3");
      return NULL;
    }

    if (PySequence_Length($input) != 3) {
      PyErr_SetString(PyExc_ValueError,"in method '" "$symname" "', argument " "$argnum" " Size mismatch. Expected 3 elements");
      return NULL;
    }

    for (int i = 0; i < 3; i++) {
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
    glm::ivec3 * temp = reinterpret_cast< glm::ivec3 * >(argp);
    $1 = *temp;
    if (SWIG_IsNewObj(res)) delete temp;
  }
}

struct ivec3 {
    
    int x, y, z;

    static length_t length();

    ivec3();
    ivec3(ivec3 const & v);
    ivec3(int scalar);
    ivec3(int s1, int s2, int s3);
    ivec3(glm::ivec2 const & a, int b);
    ivec3(int a, glm::ivec2 const & b);
    ivec3(glm::ivec4 const & v);

    /*ivec3 & operator=(ivec3 const & v);*/
};

ivec3 operator+(ivec3 const & v, int scalar);
ivec3 operator+(int scalar, ivec3 const & v);
ivec3 operator+(ivec3 const & v1, ivec3 const & v2);
ivec3 operator-(ivec3 const & v, int scalar);
ivec3 operator-(int scalar, ivec3 const & v);
ivec3 operator-(ivec3 const & v1, ivec3 const & v2);
ivec3 operator*(ivec3 const & v, int scalar);
ivec3 operator*(int scalar, ivec3 const & v);
ivec3 operator*(ivec3 const & v1, ivec3 const & v2);
ivec3 operator/(ivec3 const & v, int scalar);
ivec3 operator/(int scalar, ivec3 const & v);
ivec3 operator/(ivec3 const & v1, ivec3 const & v2);
/*ivec3 operator%(ivec3 const & v, int scalar);
ivec3 operator%(int scalar, ivec3 const & v);
ivec3 operator%(ivec3 const & v1, ivec3 const & v2);*/
bool operator==(ivec3 const & v1, ivec3 const & v2);
bool operator!=(ivec3 const & v1, ivec3 const & v2);

%extend ivec3 {

    // [] getter
    // out of bounds throws a string, which causes a Lua error
    int __getitem__(int i) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::ivec3::__getitem__()");
            }
            return (*$self)[i-1];
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::ivec3::__getitem__()");
            }
            return (*$self)[i];
        #endif
    }

    // [] setter
    // out of bounds throws a string, which causes a Lua error
    void __setitem__(int i, int f) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::ivec3::__setitem__()");
            }
            (*$self)[i-1] = f;
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::ivec3::__setitem__()");
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
    ivec3 operator+(ivec3 const & v) {return (*$self) + v;}
    ivec3 operator+(int scalar) {return (*$self) + scalar;}
    ivec3 operator-(ivec3 const & v) {return (*$self) - v;}
    ivec3 operator-(int scalar) {return (*$self) - scalar;}
    ivec3 operator*(ivec3 const & v) {return (*$self) * v;}
    ivec3 operator*(int scalar) {return (*$self) * scalar;}
    ivec3 operator/(ivec3 const & v) {return (*$self) / v;}
    ivec3 operator/(int scalar) {return (*$self) / scalar;}
    /*ivec3 operator%(ivec3 const & v) {return (*$self) % v;}
    ivec3 operator%(int scalar) {return (*$self) % scalar;}*/
    bool operator==(ivec3 const & v) {return (*$self) == v;}
    bool operator!=(ivec3 const & v) {return (*$self) != v;}
};
