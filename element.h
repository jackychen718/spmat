#pragma once


struct Element
{
	uint64_t row;
	uint64_t col;
	uint64_t val;
	Element()
	{
		this->row = 0;
		this->col = 0;
		this->val = 0;
	}

	Element(uint64_t row,int col,uint64_t val)
	{
		this->row = row;
		this->col = col;
		this->val = val;
	}
};
