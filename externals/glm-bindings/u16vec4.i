%{
#include <stdint.h>
%}

struct u16vec4 {
    
    uint16_t x, y, z, w;

    static length_t length();

    u16vec4();
    u16vec4(u16vec4 const & v);
    u16vec4(uint16_t scalar);
    u16vec4(uint16_t s1, uint16_t s2, uint16_t s3, uint16_t s4);
    u16vec4(u16vec2 const & a, u16vec2 const & b);
    u16vec4(u16vec2 const & a, uint16_t b, uint16_t c);
    u16vec4(uint16_t a, u16vec2 const & b, uint16_t c);
    u16vec4(uint16_t a, uint16_t b, u16vec2 const & c);
    u16vec4(u16vec3 const & a, uint16_t b);
    u16vec4(uint16_t a, u16vec3 const & b);

    /*u16vec4 & operator=(u16vec4 const & v);*/
};

u16vec4 operator+(u16vec4 const & v, uint16_t scalar);
u16vec4 operator+(uint16_t scalar, u16vec4 const & v);
u16vec4 operator+(u16vec4 const & v1, u16vec4 const & v2);
u16vec4 operator-(u16vec4 const & v, uint16_t scalar);
u16vec4 operator-(uint16_t scalar, u16vec4 const & v);
u16vec4 operator-(u16vec4 const & v1, u16vec4 const & v2);
u16vec4 operator*(u16vec4 const & v, uint16_t scalar);
u16vec4 operator*(uint16_t scalar, u16vec4 const & v);
u16vec4 operator*(u16vec4 const & v1, u16vec4 const & v2);
u16vec4 operator/(u16vec4 const & v, uint16_t scalar);
u16vec4 operator/(uint16_t scalar, u16vec4 const & v);
u16vec4 operator/(u16vec4 const & v1, u16vec4 const & v2);
/*u16vec4 operator%(u16vec4 const & v, uint16_t scalar);
u16vec4 operator%(uint16_t scalar, u16vec4 const & v);
u16vec4 operator%(u16vec4 const & v1, u16vec4 const & v2);*/
bool operator==(u16vec4 const & v1, u16vec4 const & v2);
bool operator!=(u16vec4 const & v1, u16vec4 const & v2);

%extend u16vec4 {

    // [] getter
    // out of bounds throws a string, which causes a Lua error
    int __getitem__(int i) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::u16vec4::__getitem__()");
            }
            return (*$self)[i-1];
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::u16vec4::__getitem__()");
            }
            return (*$self)[i];
        #endif
    }

    // [] setter
    // out of bounds throws a string, which causes a Lua error
    void __setitem__(int i, int f) throw (std::out_of_range) {
        #ifdef SWIGLUA
            if(i < 1 || i > $self->length()) {
                throw std::out_of_range("in glm::u16vec4::__setitem__()");
            }
            (*$self)[i-1] = f;
        #else
            if(i < 0 || i >= $self->length()) {
                throw std::out_of_range("in glm::u16vec4::__setitem__()");
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
    u16vec4 operator+(u16vec4 const & v) {return (*$self) + v;}
    u16vec4 operator+(uint16_t scalar) {return (*$self) + scalar;}
    u16vec4 operator-(u16vec4 const & v) {return (*$self) - v;}
    u16vec4 operator-(uint16_t scalar) {return (*$self) - scalar;}
    u16vec4 operator*(u16vec4 const & v) {return (*$self) * v;}
    u16vec4 operator*(uint16_t scalar) {return (*$self) * scalar;}
    u16vec4 operator/(u16vec4 const & v) {return (*$self) / v;}
    u16vec4 operator/(uint16_t scalar) {return (*$self) / scalar;}
    /*u16vec4 operator%(u16vec4 const & v) {return (*$self) % v;}
    u16vec4 operator%(uint16_t scalar) {return (*$self) % scalar;}*/
    bool operator==(u16vec4 const & v) {return (*$self) == v;}
    bool operator!=(u16vec4 const & v) {return (*$self) != v;}
};
