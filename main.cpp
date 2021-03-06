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

    string fn_src = "/home/tiliang/Pictures/scene.jpg";
    img = imread(fn_src, IMREAD_GRAYSCALE);

    const int threshold = 6;

    /******************************************************************************************/

    // Halide
    Halide::Buffer<uint8_t> input(img.data, img.cols, img.rows);
    input.set_name("input");

    static const int offsets16[][2] = {
        { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 }, { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 },
        { 0, -3 }, { -1, -3 }, { -2, -2 }, { -3, -1 }, { -3, 0 }, { -3, 1 }, { -2, 2 }, { -1, 3 },
        { 0, 3 }, { 1, 3 }, { 2, 2 }, { 3, 1 }, { 3, 0 }, { 3, -1 }, { 2, -2 }, { 1, -3 }
    };

    Var x("x"), y("y");
    Func clamped("clamped");
    clamped(x, y) = input(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1));

    Halide::Buffer<int> ofs_buf(offsets16);
    ofs_buf.set_name("ofs_buf");
    Var i("i");

    Func diffs("diffs");
    diffs(x, y, i) = cast<short>(clamped(x, y)) - cast<short>(clamped(x + ofs_buf(0, i), y + ofs_buf(1, i)));

    RDom r24(0, 24, "r");
    Func is_corner("isCorner"), count("count");
    count(x, y) = { 0, 0 };
    r24.where(count(x, y)[0] < 9 && count(x, y)[1] < 9);
    Expr count_below = select(diffs(x, y, r24) < -threshold, count(x, y)[0] + 1, 0);
    Expr count_above = select(diffs(x, y, r24) > threshold, count(x, y)[1] + 1, 0);
    count(x, y) = { count_below, count_above };
    is_corner(x, y) = count(x, y)[0] >= 9 || count(x, y)[1] >= 9;

    Func min_max_vals("min_max_vals");
    RDom ri(0, 9, "ri");
    min_max_vals(x, y, i) = { maximum(diffs(x, y, i + ri)), minimum(diffs(x, y, i + ri)) };

    Region bounds = { { 3, input.width() - 6 }, { 3, input.height() - 6 } };
    Func score_ub("score_unbounded"), score("score");
    RDom ro(0, 16, "ro");
    Expr score_val = max(-minimum(min_max_vals(x, y, ro)[0]), maximum(min_max_vals(x, y, ro)[1])) - 1;
    score_ub(x, y) = select(is_corner(x, y), score_val, 0);
    score(x, y) = BoundaryConditions::constant_exterior(score_ub, 0, bounds)(x, y);

    Func result_ub("result_unbounded"), result("result");
    result_ub(x, y) = score(x, y) > max(threshold - 1, score(x - 1, y), score(x + 1, y),
                          score(x - 1, y - 1), score(x, y - 1), score(x + 1, y - 1),
                          score(x - 1, y + 1), score(x, y + 1), score(x + 1, y + 1));
    result(x, y) = BoundaryConditions::constant_exterior(
        result_ub, false, bounds)(x, y);

    // schedule
    Var yo("yo"), yi("yi");
    const int vec_size_short = get_host_target().natural_vector_size<short>();
    result.split(y, yo, yi, 32).vectorize(x, vec_size_short);
    score.compute_at(result, yi).store_at(result, yo).vectorize(x, vec_size_short);

    is_corner.compute_root();

    // test
    //    result.print_loop_nest();
    //    result.compile_to_lowered_stmt("isCorner.html", {}, HTML);
    Buffer<bool> output(input.width(), input.height());
    result.realize(output);

    vector<Point> result_hld;
    for (int r = 0; r < output.height(); ++r) {
        for (int c = 0; c < output.width(); ++c) {
            int x = c, y = r;
            if (output(x, y))
                result_hld.emplace_back(x, y);
        }
    }

    //    Buffer<bool> output(9, 9);
    //    output.set_min(46, 46);
    //    is_corner.realize(output);
    //    for (int y = 0; y < 9; ++y) {
    //        for (int x = 0; x < 9; ++x)
    //            cout << output(x + 46, y + 46) << " ";
    //        cout << endl;
    //    }

    //    RDom r(0, 24, "r");
    //    Func isCorner("isCorner"), count("count");
    //    count(x, y) = { 0, 0 };
    //    r.where(count(x, y)[0] < 9 && count(x, y)[1] < 9);
    //    Expr count_below = select(diffs(x, y, r) < -6, count(x, y)[0] + 1, 0);
    //    Expr count_above = select(diffs(x, y, r) > 6, count(x, y)[1] + 1, 0);
    //    count(x, y) = { count_below, count_above };
    //    isCorner(x, y) = count(x, y)[0] >= 9 || count(x, y)[1] >= 9;

    /******************************************************************************************/

#if 1
    vector<KeyPoint> result_ocv;
    // benchmark
    double time_hld = Tools::benchmark(3, 3, [&] {
        result.realize(output);
    });
    double time_ocv = Tools::benchmark(3, 10, [&] {
        FAST(img, result_ocv, threshold, true);
    });
    printf("ocv: pts[%zu] time[%f ms]\n", result_ocv.size(), time_ocv * 1e3);
    printf("hld: pts[%zu] time[%f ms]\n", result_hld.size(), time_hld * 1e3);
    cout.flush();
    Mat img_ocv, img_hld;
    cvtColor(img, img_ocv, COLOR_GRAY2BGR);
    img_hld = img_ocv.clone();
    for (const auto& kp : result_ocv)
        img_ocv.at<Vec3b>(kp.pt) = { 0, 0, 255 };
    for (const auto& pt : result_hld)
        img_hld.at<Vec3b>(pt) = { 0, 0, 255 };
    imshow("src", img);
    imshow("ocv", img_ocv);
    imshow("hld", img_hld);
    imshow("diff", (img_hld - img_ocv) > 0);
    waitKey(0);
#endif
    return 0;
}
