#include "slider.h"
#include "../common/logger/logger.h"

using Executor = std::function<void()>;
using Executors = std::vector<Executor>;

Slider::Slider(Executors executors)
  :executors(executors)
{
  size = executors.size();
}

void Slider::next()
{
  currentIndex = currentIndex == size - 1 ? 0 : currentIndex + 1;
  DEBUG("Next");
}

void Slider::previous()
{
  currentIndex = currentIndex == 0 ? size - 1 : currentIndex - 1;
  DEBUG("Previous");
}

void Slider::exec()
{
  executors[currentIndex]();
}
