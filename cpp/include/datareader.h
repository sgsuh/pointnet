#pragma once

#include <iostream>

namespace ncnn
{
// date read wrapper
class DataReader
{
public:
    DataReader();
    virtual ~DataReader();

    // parse plain param text
    // return 1 if scan success
    virtual int scan(const char* format, void* p) const;

    // read binary param and model data
    // return bytes read
    virtual size_t read(void* buf, size_t size) const;

    // get model data reference
    // return bytes referenced
    virtual size_t reference(size_t size, const void** buf) const;
};

class DataReaderFromStdioPrivate;

class DataReaderFromStdio : public DataReader
{
public:
    explicit DataReaderFromStdio(FILE* fp);
    virtual ~DataReaderFromStdio();

    virtual int scan(const char* format, void* p) const;

    virtual size_t read(void* buf, size_t size) const;

private:
    DataReaderFromStdioPrivate* const d;
};
}