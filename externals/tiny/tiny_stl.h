#pragma once

#include <string>
#include <fstream>
#include <iostream>

class v3
{
    public:
    v3(char* facet) {
        char f1[4] = {facet[0],
            facet[1],facet[2],facet[3]};

        char f2[4] = {facet[4],
            facet[5],facet[6],facet[7]};

        char f3[4] = {facet[8],
            facet[9],facet[10],facet[11]};

        m_x = *((float*) f1 );
        m_y = *((float*) f2 );
        m_z = *((float*) f3 );
    }
    
    float m_x, m_y, m_z;
};

inline bool read_stl(std::string fname, std::vector<float> &p, std::vector<float> &n){
    using namespace std;
    
    struct stat st;
    stat(fname.c_str(), &st);
    auto size = st.st_size;


    //!!
    //don't forget ios::binary
    //!!
    ifstream myFile (fname.c_str(), ios::in | ios::binary);

    char header_info[80] = "";
    char nTri[4];
    unsigned long nTriLong;

    //read 80 byte header
    if (myFile) {
        myFile.read(header_info, 80);
        // cout <<"header: " << header_info << endl;
    }
    else{
        // cout << "error" << endl;
        return false;
    }

    //read 4-byte ulong
    if (myFile) {
        myFile.read (nTri, 4);
        nTriLong = *((unsigned long*)nTri) ;
        // cout <<"n Tri: " << nTriLong << endl;
    }
    else{
        // cout << "error" << endl;
        return false;
    }

    const size_t facetSize = 3*sizeof(float_t) + 3*3*sizeof(float_t) + sizeof(uint16_t);
    if (size != (84 + (nTriLong * facetSize))) {
        std::cout<<"Error, currently only binary STL files are supported."<<std::endl;
        return false;
    }

    n.resize(3 * 3 * nTriLong);
    p.resize(3 * 3 * nTriLong);

    //now read in all the triangles
    for(unsigned int i = 0; i < nTriLong; i++){

        char facet[50];

        if (myFile) {

            //read one 50-byte triangle
            myFile.read (facet, 50);

            //populate each point of the triangle
            //using v3::v3(char* bin);
    
            //facet + 12 skips the triangle's unit normal
            v3 norm(facet);
            v3 p1(facet+12);
            v3 p2(facet+24);
            v3 p3(facet+36);
            
            //add a new triangle to the array
            n[i * 9 + 0 * 3 + 0] = n[i * 9 + 1 * 3 + 0] = n[i * 9 + 2 * 3 + 0] = norm.m_x;
            n[i * 9 + 0 * 3 + 1] = n[i * 9 + 1 * 3 + 1] = n[i * 9 + 2 * 3 + 1] = norm.m_y;
            n[i * 9 + 0 * 3 + 2] = n[i * 9 + 1 * 3 + 2] = n[i * 9 + 2 * 3 + 2] = norm.m_z;

            p[i * 9 + 0 * 3 + 0] = p1.m_x;
            p[i * 9 + 0 * 3 + 1] = p1.m_y;
            p[i * 9 + 0 * 3 + 2] = p1.m_z;
            p[i * 9 + 1 * 3 + 0] = p2.m_x;
            p[i * 9 + 1 * 3 + 1] = p2.m_y;
            p[i * 9 + 1 * 3 + 2] = p2.m_z;
            p[i * 9 + 2 * 3 + 0] = p3.m_x;
            p[i * 9 + 2 * 3 + 1] = p3.m_y;
            p[i * 9 + 2 * 3 + 2] = p3.m_z;
        } 
    }

    return true;

}
