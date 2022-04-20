SCRIPT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPTPATH=`dirname $SCRIPT`
echo $SCRIPTPATH


g++ -std=c++11 ../codes/render_balls_so.cpp -o ../codes/render_balls_so.so -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -c
