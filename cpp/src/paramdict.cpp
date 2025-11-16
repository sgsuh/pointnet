#include "paramdict.h"

#include "mat.h"
#include "datareader.h"

namespace ncnn
{
class ParamDictPrivate
{
public:
    struct
    {
        // 0 = null
        // 1 = int / float
        // 2 = int
        // 3 = float
        // 4 = array of int / float
        // 5 = array of int
        // 6 = array of float
        int type;

        union
        {
            int i;
            float f;
        };

        Mat v;
    } params[NCNN_MAX_PARAM_COUNT];
};

ParamDict::ParamDict()
    : d(new ParamDictPrivate)
{
    clear();
}

ParamDict::~ParamDict()
{
    delete d;
}

void ParamDict::clear()
{
    for(int i = 0; i < NCNN_MAX_PARAM_COUNT; i++) {
        d->params[i].type = 0;
        d->params[i].v = Mat();
    }
}

int ParamDict::get(int id, int def) const
{
    return d->params[id].type ? d->params[id].i : def;
}

float ParamDict::get(int id, float def) const
{
    return d->params[id].type ? d->params[id].f : def;
}

Mat ParamDict::get(int id, const Mat& def) const
{
    return d->params[id].type ? d->params[id].v : def;
}

void ParamDict::set(int id, int i)
{
    d->params[id].type = 2;
    d->params[id].i = i;
}

void ParamDict::set(int id, float f)
{
    d->params[id].type = 3;
    d->params[id].f = f;
}

void ParamDict::set(int id, const Mat& v)
{
    d->params[id].type = 4;
    d->params[id].v = v;
}

static bool vstr_is_float(const char vstr[16])
{
    // look ahead for determine isfloat
    for(int j = 0; j < 16; j++) {
        if(vstr[j] == '\0') {
            break;
        }

        if(vstr[j] == '.' || tolower(vstr[j]) == 'e') {
            return true;
        }
    }

    return false;
}

static float vstr_to_float(const char vstr[16])
{
    double v = 0.0;

    const char* p = vstr;

    // sign
    bool sign = *p != '-';

    if(*p == '+' || *p == '-') {
        p++;
    }

    // digits before decimal point or exponent
    unsigned int v1 = 0;

    while(isdigit(*p)) {
        v1 = v1 * 10 + (*p - '0');
        
        p++;
    }

    v = (double)v1;

    // digits after decimal point
    if(*p == '.') {
        p++;

        unsigned int pow10 = 1;
        unsigned int v2 = 0;

        while(isdigit(*p)) {
            v2 = v2 * 10 + (*p - '0');
            pow10 *= 10;
            p++;
        }

        v += v2 / (double)pow10;
    }

    // exponent
    if(*p == 'e' || *p == 'E') {
        p++;

        // sign of exponent
        bool fact = *p != '-';

        if(*p == '+' || *p == '-') {
            p++;
        }

        // digits of exponent
        unsigned int expon = 0;

        while(isdigit(*p)) {
            expon = expon * 10 + (*p - '0');

            p++;
        }

        double scale = 1.0;

        while(expon >= 8) {
            scale *= 1e8;
            expon -= 8;
        }

        while(expon > 0) {
            scale *= 10.0;
            expon -= 1;
        }

        v = fact ? v * scale : v / scale;
    }

    return sign ? (float)v : (float)-v;
}

int ParamDict::load_param(const DataReader& dr)
{
    clear();

    // parse each key = value pair
    int id = 0;

    while(dr.scan("%d=", &id) == 1) {
        bool is_array = id <= -23300;

        if(is_array) {
            id = -id - 23300;
        }

        if(id >= NCNN_MAX_PARAM_COUNT) {
            printf("id < NCNN_MAX_PARAM_COUNT failed (id=%d, NCNN_MAX_PARAM_COUNT=%d)\n", id, NCNN_MAX_PARAM_COUNT);

            return -1;
        }

        if(is_array) {
            int len = 0;
            int nscan = dr.scan("%d", &len);

            if(nscan != 1) {
                printf("ParamDict read array length failed\n");

                return -1;
            }

            d->params[id].v.create(len);

            for(int j = 0; j < len; j++) {
                char vstr[16];

                nscan = dr.scan(",%15[^,\n ]", vstr);

                if(nscan != 1) {
                    printf("ParamDict read array element failed\n");

                    return -1;
                }

                bool is_float = vstr_is_float(vstr);

                if(is_float) {
                    float* ptr = d->params[id].v;

                    ptr[j] = vstr_to_float(vstr);
                }
                else {
                    int* ptr = d->params[id].v;

                    nscan = sscanf(vstr, "%d", &ptr[j]);

                    if(nscan != 1) {
                        printf("ParamDict parse array element failed\n");
                        
                        return -1;
                    }
                }

                d->params[id].type = is_float ? 6 : 5;
            }
        }
        else {
            char vstr[16];
            int nscan = dr.scan("%15s", vstr);

            if(nscan != 1) {
                printf("ParamDict read value failed\n");

                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if(is_float) {
                d->params[id].f = vstr_to_float(vstr);
            }
            else {
                nscan = sscanf(vstr, "%d", &d->params[id].i);

                if(nscan != 1) {
                    printf("ParamDict parse value failed\n");

                    return -1;
                }
            }

            d->params[id].type = is_float ? 3 : 2;
        }
    }

    return 0;
}
}