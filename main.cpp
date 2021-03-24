#include "Halide.h"
#include <halide_benchmark.h>
#include <halide_image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace Halide;
using namespace std;
using namespace cv;

int main()
{
    Mat img(500, 500, CV_8UC1, 100);
    Point pt = { 50, 50 };
    {
        srand(0);
        vector<Point> ofs = {
            { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 }, { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 },
            { 0, -3 }, { -1, -3 }, { -2, -2 }, { -3, -1 }, { -3, 0 }, { -3, 1 }, { -2, 2 }, { -1, 3 }
        };
        for (int i = 0; i < 16; ++i) {
            img.at<uchar>(pt + ofs[i]) = 50 + rand() % 10;
        }
        const int start_idx = 3;
        for (int i = start_idx; i < start_idx + 9; ++i) {
            img.at<uchar>(pt + ofs[i]) = 150 + rand() % 10;
        }
    }
    imshow("src", img);

    /******************************************************************************************/

    // Halide
    Halide::Buffer<uint8_t> input(img.data, img.cols, img.rows);
    input.set_name("input");

    static const int offsets16[][2] = {
        { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 }, { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 },
        { 0, -3 }, { -1, -3 }, { -2, -2 }, { -3, -1 }, { -3, 0 }, { -3, 1 }, { -2, 2 }, { -1, 3 },
        { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 }, { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 }
    };

    Func clamped("clamped");
    clamped = BoundaryConditions::repeat_edge(input);

    Halide::Buffer<int> ofs_buf(offsets16);
    ofs_buf.set_name("ofs_buf");
    Var i("i");

    Func diffs("nbs");
    Var x("x"), y("y");
    diffs(x, y, i) = cast<short>(clamped(x, y)) - cast<short>(clamped(x + ofs_buf(0, i), y + ofs_buf(1, i)));

    RDom r(0, 24, "r");
    Func isCorner("isCorner"), count("count");
    count(x, y) = { 0, 0 };
    r.where(count(x, y)[0] < 9 && count(x, y)[1] < 9);
    Expr count_below = select(diffs(x, y, r) < -6, count(x, y)[0] + 1, 0);
    Expr count_above = select(diffs(x, y, r) > 6, count(x, y)[1] + 1, 0);
    count(x, y) = { count_below, count_above };
    isCorner(x, y) = count(x, y)[0] >= 9 || count(x, y)[1] >= 9;

    isCorner.print_loop_nest();
    isCorner.compile_to_lowered_stmt("isCorner.html", {}, HTML);

    Buffer<bool> output(9, 9);
    output.set_min(46, 46);
    isCorner.realize(output);
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 9; ++x)
            cout << output(x + 46, y + 46) << " ";
        cout << endl;
    }

    Func t("t");
    t(i) = diffs(50, 50, i);
    t.trace_stores();
    t.realize(24);

    /******************************************************************************************/

    vector<KeyPoint> kps;
    FAST(img, kps, 6, true);
    cout << "kps: " << kps.size() << endl;
    Mat img_kps;
    cvtColor(img, img_kps, COLOR_GRAY2BGR);
    for (const auto& kp : kps)
        img_kps.at<Vec3b>(kp.pt) = { 0, 0, 255 };
    imshow("kps", img_kps);
    waitKey(0);
    return 0;
}
