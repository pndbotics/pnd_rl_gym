#pragma once

#include <array>
#include <cmath>
#include <cstdint>

class KeyMap
{
public:
    static constexpr int A      = 0;
    static constexpr int B      = 1;
    static constexpr int X      = 2;
    static constexpr int Y      = 3;
    static constexpr int LB     = 4;
    static constexpr int RB     = 5;
    static constexpr int select = 6;
    static constexpr int start  = 7;
    static constexpr int home   = 8;
    static constexpr int lo     = 9;
    static constexpr int ro     = 10;
};

class RemoteController
{
public:
    RemoteController()
        : lx(0), ly(0), rx(0), ry(0),
          lt(0), rt(0), xx(0), yy(0),
          button{},
          dead_area(5000),
          max_value(32767),
          ly_dir(-1.0f),
          lx_dir(-1.0f),
          rx_dir(-1.0f),
          max_speed_x(1.0f),
          min_speed_x(-1.0f),
          max_speed_y(1.0f),
          min_speed_y(-1.0f),
          max_speed_yaw(1.0f),
          min_speed_yaw(-1.0f)
    {
        button.fill(0);
    }

    void set(const std::array<float, 19>& data)
    {
        for (int i = 0; i < 10; ++i)
        {
            button[i] = data[i + 8];
        }

        lx = data[0];
        ly = data[1];
        rx = data[2];
        ry = data[3];
        lt = data[4];
        rt = data[5];
        xx = data[6];
        yy = data[7];
    }

    float get_walk_x_direction_speed() const
    {
        if (lt < 1000)
        {
            int32_t x_value = ly;
            int32_t abs_x_value = std::abs(x_value);

            if (abs_x_value > dead_area && abs_x_value <= max_value)
            {
                float scale = static_cast<float>(abs_x_value - dead_area) /
                              static_cast<float>(max_value - dead_area);

                if (x_value > 0)
                    return ly_dir * max_speed_x * scale;
                else
                    return ly_dir * min_speed_x * scale;
            }
            else if (abs_x_value <= 1)
            {
                return static_cast<float>(ly) * ly_dir;
            }
            else
            {
                return 0.0f;
            }
        }
        else
        {
            return 0.70f;
        }
    }

    float get_walk_y_direction_speed() const
    {
        int32_t y_value = lx;
        int32_t abs_y_value = std::abs(y_value);

        if (abs_y_value > dead_area && abs_y_value <= max_value)
        {
            float scale = static_cast<float>(abs_y_value - dead_area) /
                          static_cast<float>(max_value - dead_area);

            if (y_value > 0)
                return lx_dir * max_speed_y * scale;
            else
                return lx_dir * min_speed_y * scale;
        }
        else
        {
            return 0.0f;
        }
    }

    float get_walk_yaw_direction_speed() const
    {
        int32_t yaw_value = rx;
        int32_t abs_yaw_value = std::abs(yaw_value);

        if (abs_yaw_value > dead_area && abs_yaw_value <= max_value)
        {
            float scale = static_cast<float>(abs_yaw_value - dead_area) /
                          static_cast<float>(max_value - dead_area);

            if (yaw_value > 0)
                return rx_dir * max_speed_yaw * scale;
            else
                return rx_dir * min_speed_yaw * scale;
        }
        else
        {
            return 0.0f;
        }
    }

public:
    int32_t lx;
    int32_t ly;
    int32_t rx;
    int32_t ry;
    int32_t lt;
    int32_t rt;
    int32_t xx;
    int32_t yy;

    std::array<int32_t, 10> button;

    int32_t dead_area;
    int32_t max_value;

    float ly_dir;
    float lx_dir;
    float rx_dir;

    float max_speed_x;
    float min_speed_x;
    float max_speed_y;
    float min_speed_y;
    float max_speed_yaw;
    float min_speed_yaw;
};
