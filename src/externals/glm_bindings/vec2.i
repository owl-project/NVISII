// glm::vec2 bindings
// 2018 Dan Wilcox <danomatika@gmail.com>

// ----- detail/type_vec2.hpp -----
%typemap(in) glm::vec2 (void *argp = 0, int res = 0) {
  int res = SWIG_ConvertPtr($input, &argp, $descriptor(glm::vec2*), $disown | 0);
  if (!SWIG_IsOK(res)) 
  { 
    if (!PySequence_Check($input)) {
      PyErr_SetString(PyExc_ValueError, "in method '" "$symname" "', argument " "$argnum" " Expected either a sequence or vec2");
      return NULL;
    }

    if (PySequence_Length($input) != 2) {
      PyErr_SetString(PyExc_ValueError,"in method '" "$symname" "', argument " "$argnum" " Size mismatch. Expected 2 elements");
      return NULL;
    }

    for (int i = 0; i < 2; i++) {
      PyObject *o = PySequence_GetItem($input,i);
      if (PyNumber_Check(o)) {
        $1[i] = (float) PyFloat_AsDouble(o);
      } else {
        PyErr_SetString(PyExc_ValueError,"in method '" "$symname" "', argument " "$argnum" " Sequence elements must be numbers");      
        return NULL;
      }
    }
  }   
  else {
    glm::vec2 * temp = reinterpret_cast< glm::vec2 * >(argp);
    $1 = *temp;
    if (SWIG_IsNewObj(res)) delete temp;
  }
}

struct vec2 {

    float x, y;

    static length_t length();

    vec2();
    vec2(vec2 const & v);
    vec2(float scalar);
    vec2(float s1, float s2);
    vec2(glm::vec3 const & v);
    vec2(glm::vec4 const & v);

    /*vec2 & operator=(vec2 const & v);*/
};

vec2 operator+(vec2 const & v, float scalar);
vec2 operator+(float scalar, vec2 const & v);
vec2 operator+(vec2 const & v1, vec2 const & v2);
vec2 operator-(vec2 const & v, float scalar);
vec2 operator-(float scalar, vec2 const & v);
vec2 operator-(vec2 const & v1, vec2 const & v2);
vec2 operator*(vec2 const & v, float scalar);
vec2 operator*(float scalar, vec2 const & v);
vec2 operator*(vec2 const & v1, vec2 const & v2);
vec2 operator/(vec2 const & v, float scalar);
vec2 operator/(float scalar, vec2 const & v);
vec2 operator/(vec2 const & v1, vec2 const & v2);
/*vec2 operator%(vec2 const & v, int scalar);
vec2 operator%(int scalar, vec2 const & v);
vec2 operator%(vec2 const & v1, vec2 const & v2);*/
bool operator==(vec2 const & v1, vec2 const & v2);
bool operator!=(vec2 const & v1, vec2 const & v2);

%extend vec2 {
    
    // [] getter
    // out of bounds throws a string, which causes a Lua error
    float __getitem__(int i) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::vec2::__getitem__()");
            }
            return (*$self)[i-1];
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::vec2::__getitem__()");
            }
            return (*$self)[i];
        #endif
    }

    // [] setter
    // out of bounds throws a string, which causes a Lua error
    void __setitem__(int i, float f) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::vec2::__setitem__()");
            }
            (*$self)[i-1] = f;
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::vec2::__setitem__()");
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
    vec2 operator+(vec2 const & v) {return (*$self) + v;}
    vec2 operator+(float scalar) {return (*$self) + scalar;}
    vec2 operator-(vec2 const & v) {return (*$self) - v;}
    vec2 operator-(float scalar) {return (*$self) - scalar;}
    vec2 operator*(vec2 const & v) {return (*$self) * v;}
    vec2 operator*(float scalar) {return (*$self) * scalar;}
    vec2 operator/(vec2 const & v) {return (*$self) / v;}
    vec2 operator/(float scalar) {return (*$self) / scalar;}
    /*vec2 operator%(vec2 const & v) {return (*$self) % v;}
    vec2 operator%(int scalar) {return glm::vec2((int)((*$self)[0]) % (int)scalar,(int)((*$self)[1]) % (int)scalar);}*/
    bool operator==(vec2 const & v) {return (*$self) == v;}
    bool operator!=(vec2 const & v) {return (*$self) != v;}
};
