// glm::ivec4 bindings

// ----- detail/type_ivec4.hpp -----

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
