#include "manager.hpp"

#include <boost/math/special_functions/round.hpp>
#include <boost/math/special_functions/modf.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

using namespace ugproj;
namespace math = boost::math;

OpticalFlowManager::OpticalFlowManager(int flowWidth, int flowHeight) :
    flowWidth(flowWidth), flowHeight(flowHeight) {};

void OpticalFlowManager::append(OpticalFlowManager::flow_t* flow) {
    flowVect.push_back(flow);
}

const cv::Vec2d OpticalFlowManager::getFlowAt(const temp_idx_t startTempIndex,
                                              const temp_idx_t endTempIndex,
                                              int x, int y) const {
    if (startTempIndex >= endTempIndex) {
        throw "startTempIndex should be before endTempIndex";
    }

    temp_idx_t tempPos = startTempIndex;
    flow_t* flow = flowVect[tempPos];
    OptFlowArray* vx = flow->first;
    OptFlowArray* vy = flow->second;
    double dx = vx->At(x, y, 0);
    double dy = vy->At(x, y, 0);
    ++tempPos;

    double _x;
    double _y;
    int _x_i;
    int _y_i;
    double _x_f;
    double _y_f;
    Eigen::RowVector2d mat_x;
    Eigen::Matrix2d mat_flow;
    Eigen::Vector2d mat_y;
    while (tempPos < endTempIndex) {
        /*
         * Bilinear Interpolation
         * 
         * +----------------+ 
         * |(_x_i, _y_i)    |(_x_i + 1, _y_i)
         * |                |
         * |                |
         * |        +       |
         * |         (_x, _y) = (_x_i + _x_f, _y_i + _y_f)
         * |                |
         * |                |
         * +----------------+
         *  (_x_i, _y_i + 1) (_x_i + 1, _y_i + 1)
         *
         *
         * Matrix operation below is from Wikipedia:
         * http://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_Square
         */
        _x = x + dx;
        _y = y + dy;
        _x_f = math::modf(_x, &_x_i);
        _y_f = math::modf(_y, &_y_i);

        mat_x << (1 - _x_f), _x_f;
        mat_flow << vx->At(_x_i, _y_i, 0), vx->At(_x_i, _y_i + 1, 0),
                    vx->At(_x_i + 1, _y_i, 0), vx->At(_x_i + 1, _y_i + 1, 0);
        mat_y << (1 - _y_f),
                 _y_f;

        dx += mat_x * mat_flow * mat_y;
        mat_flow << vy->At(_x_i, _y_i, 0), vy->At(_x_i, _y_i + 1, 0),
                    vy->At(_x_i + 1, _y_i, 0), vy->At(_x_i + 1, _y_i + 1, 0);
        dy += mat_x * mat_flow * mat_y;
    }

    return cv::Vec2d(dx, dy);
}
