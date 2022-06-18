#include <utils.h>
#include <ctime>
#include <sys/time.h>

//获取当前UTC格式时间,单位:秒
double getLocalTimeToUTC()
{
    struct timeval begin;
    gettimeofday(&begin,NULL);
    double time = (double)begin.tv_sec +
                  (double)begin.tv_usec/1000000;
    return time;
}
std::string colouredString(std::string str, std::string colour, std::string option)
{
  double time_now = getLocalTimeToUTC();
  std::string time_string = std::to_string(time_now);
  return "[" + time_string + "]: " + option + colour + str + RESET;
}
