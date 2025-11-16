#include "datareader.h"

namespace ncnn
{
DataReader::DataReader()
{

}

DataReader::~DataReader()
{

}

int DataReader::scan(const char*, void*) const
{
    return 0;
}

size_t DataReader::read(void*, size_t) const
{
    return 0;
}

size_t DataReader::reference(size_t, const void**) const
{
    return 0;
}

class DataReaderFromStdioPrivate
{
public:
    DataReaderFromStdioPrivate(FILE* _fp)
        : fp(_fp)
    {

    }

    FILE* fp;
};

DataReaderFromStdio::DataReaderFromStdio(FILE* _fp)
    : DataReader(), d(new DataReaderFromStdioPrivate(_fp))
{

}

DataReaderFromStdio::~DataReaderFromStdio()
{
    delete d;
}

int DataReaderFromStdio::scan(const char* format, void* p) const
{
    return fscanf(d->fp, format, p);
}

size_t DataReaderFromStdio::read(void* buf, size_t size) const
{
    return fread(buf, 1, size, d->fp);
}
}