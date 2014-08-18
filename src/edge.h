#ifndef EDGE_H
#define EDGE_H

using namespace std;

class Edge {

public:
  int u, v;

public:

  bool operator< (const Edge& e) const {
    bool result = false;
    if(u<e.u || (u==e.u && v<e.v)) {
      result = true;
    }
    return result;
  }
  bool operator	== (const Edge&e) const{
	  bool result = false;
	  if (u == e.u && v == e.v){
		  result = true;
	  }
	  return result;
  }

  std::pair<int, int> pair() const {
    return std::pair<int, int>(u, v);
  }

  Edge(int u_, int v_){
	  if (u_ < v_){
		  u = u_;
		  v = v_;
	  } else{
		  u = v_;
		  v = u_;
	  }
  }

  string to_string(){
	  std::string s;
	  std::stringstream out;
	  out<<"("<<u<<","<<v<<")";
	  return out.str();
  }
};

#endif
