/// class ImageRectangle -

#ifndef SSV_IMAGERECTANGLE_H
#define SSV_IMAGERECTANGLE_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <opencv2/core/types.hpp>
#include "Segment.h"

/**
 * ÐºÐ»Ð°ÑÑ Ð´Ð»Ñ Ð¾Ð¿Ð¸ÑÐ°Ð½Ð¸Ñ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¾Ð¹  Ð¾Ð±Ð»Ð°ÑÑ‚Ð¸ Ð½Ð° Ð¸Ð·Ð¾Ð±Ñ€Ð°Ð¶ÐµÐ½Ð¸Ð¸
 * */

//class ConvexTetragon;

template<class T>
class ImageRectangle {

public:

/*
    struct Segment{
        Segment(Eigen::Vector2d _pt1, Eigen::Vector2d _pt2): pt1(_pt1), pt2(_pt2){}
        Eigen::Vector2d pt1, pt2;
    };
*/

    enum RectangleEdge { minXEdge, maxXEdge, minYEdge, maxYEdge};

                                        ImageRectangle          ( );
                                        ImageRectangle          ( T x_, T y_);
                                        ImageRectangle          ( T x_, T y_, T width_, T height_);
                                        ImageRectangle          ( T x_, T y_, T width_, T height_, T border_);
                                        ImageRectangle          ( Eigen::Matrix<T, 2, 1>  ul, Eigen::Matrix<T, 2, 1>  br );
                                        ImageRectangle          ( Eigen::Matrix<T, 2, 1>  pt1, Eigen::Matrix<T, 2, 1>  pt2, Eigen::Matrix<T, 2, 1>  pt3, Eigen::Matrix<T, 2, 1>  pt4 );

    /** geometric operations*/

    void                                overlap_inPlace         ( const ImageRectangle<T> &rect );
    ImageRectangle                      overlap                 ( const ImageRectangle<T> &rect );


    void                                circumscribe_inPlace    ( const ImageRectangle<T> &rect);
    ImageRectangle                      circumscribe            ( const ImageRectangle<T> &rect );

    ImageRectangle                      rectBetween             ( const ImageRectangle<T> & rect1, const ImageRectangle<T> & rect2, ImageRectangle<T> *remainder, RectangleEdge edge);
    std::vector<ImageRectangle>         subtract(const ImageRectangle<T> &rect);
    std::vector<ImageRectangle>         subdivide               ( int level );

    void                                slice                   ( ImageRectangle<T> *pRectSlice, ImageRectangle<T> *pRectRemainder, T amount, RectangleEdge edge);


    ImageRectangle                      maxSizeTo               ( const ImageRectangle<T> &rect );
    ImageRectangle                      minSizeTo               ( const ImageRectangle<T> &rect );
    void                                fitSizeTo               ( const ImageRectangle<T> &rect );
    void                                scaleSizeTo             ( T scale );
    void                                fitSizeTo               ( T new_width, T new_height );
    void                                cutOutsidePoints        ( const Eigen::Matrix2Xd &points_in, Eigen::Matrix2Xd &points_out);
    void                                pushInside              ( ImageRectangle<T> rect, Eigen::Matrix<T, 2, 1> *shift = nullptr);
    void                                pushOutside             ( ImageRectangle<T> &rect  );
    Eigen::Matrix<T, 2, 1>              closestVectorFromPoint  ( Eigen::Matrix<T, 2, 1>  pt);
    T                                   calc_area               () const;

    template<typename T2>
    bool                                check_hasInside         (Eigen::Matrix<T2, 2, 1> pt) const;
    bool                                check_hasInside         ( ImageRectangle<T> rect) const;
    bool                                check_isEmpty           ( ) const;


    /** set methods*/
    void                                set                     (  T x_, T y_, T width_, T height_, T border_);
    void                                set                     (  T x_, T y_, T width_, T height_);
    void                                set                     (  T x_, T y_);
    void                                set_size                (  T width_, T height_);
    void                                set_ul                  (  const Eigen::Matrix<T, 2, 1> &ul);

    /**get methods */
    T                                   calc_maxX() const;
    T                                   calc_maxY() const;

    void                                get                     ( T &x_, T &y_, T &width_, T &height_, T &border) const;
    void                                get                     ( T &x_, T &y_, T &width_, T &height_) const;
    void                                get                     ( T& x_, T& y_) const;
    cv::Rect_<T>                        get_OpencV              ( )     const;
    cv::Point_<T>                       get_ulOpenCV            ( )     const;
    cv::Point_<T>                       get_brOpenCV            ( )     const;
    cv::Point_<T>                       get_whOpenCV            ( )     const;
    cv::Size_<T>                        get_whSizeCV            ( )     const;
    Eigen::Matrix<T, 2, 1>              get_ul                  ( )     const;
    Eigen::Matrix<T, 2, 1>              get_ur                  ( )     const;
    Eigen::Matrix<T, 2, 1>              get_br                  ( )     const;
    Eigen::Matrix<T, 2, 1>              get_bl                  ( )     const;
    Eigen::Matrix<T, 2, 1>              get_wh                  ( )     const;
    Eigen::Matrix<T, 2, 4>              get_corners             ( )     const;
    Eigen::Matrix<T, 2, 1>              get_center              ( )     const;
    void                                get_regularGrid         ( Eigen::Vector2i divs, Eigen::Matrix<double, 2, Eigen::Dynamic> &out) const ;

//    void                              get_ul              ( Eigen::Matrix<T, 2, 1> &up_left_corner) const;
    void                                get_border              ( T& border_ ) const;
    void                                get_size                ( T &width_, T &height_ ) const;
    Segment                             get_sideSegment         ( RectangleEdge edge);
    std::vector<Segment>                get_sidesList           () const;

    void                                shrinkBy                ( T factor);
    void                                shiftBy                 ( T x, T y);
    void                                shiftBy                 ( Eigen::Matrix<T, 2, 1> v);
    void                                set_ul2zero();

//    ImageRectangle                      cast                    (  ){ };

    void                                print                   (   );


    template<typename newT >
    operator ImageRectangle< newT >() const
    {
        ImageRectangle< newT > result;

        result.x = x;
        result.y = y;
        result.width = width;
        result.height = height;

        return result;
    }


    T           x,  y;
    T           width,   height;
    T           border;
};


typedef  ImageRectangle<int>            ImageRectangleI;
typedef  ImageRectangle<float>          ImageRectangleF;
typedef  ImageRectangle<double>         ImageRectangleD;




template <typename  T>
ImageRectangle<T>::ImageRectangle(): x(0), y(0), width(0), height(0) { }

template <typename  T>
ImageRectangle<T>::ImageRectangle(T x_, T y_) : x(x_), y(y_){}

template <typename  T>
ImageRectangle<T>::ImageRectangle(T x_, T y_, T width_, T height_): x(x_), y(y_), width(width_), height(height_) { }

template <typename  T>
ImageRectangle<T>::ImageRectangle(T x_, T y_, T width_, T height_, T border_): x(x_), y(y_), width(width_), height(height_), border(border_) { }


template<class T>
ImageRectangle<T>::ImageRectangle(Eigen::Matrix<T, 2, 1> ul, Eigen::Matrix<T, 2, 1> br) {

    x           = ul(0,0);
    y           = ul(1,0);

    width       = br(0,0) - ul(0,0);
    height      = br(1,0) - ul(1,0);
}

template<class T>
ImageRectangle<T>::ImageRectangle(Eigen::Matrix<T, 2, 1> pt1, Eigen::Matrix<T, 2, 1> pt2, Eigen::Matrix<T, 2, 1> pt3, Eigen::Matrix<T, 2, 1> pt4) {

    x           =   std::min( std::min( pt1(0,0), pt2(0,0) ), std::min( pt3(0,0), pt4(0,0) ) );
    y           =   std::min( std::min( pt1(1,0), pt2(1,0) ), std::min( pt3(1,0), pt4(1,0) ) );

    width       =   std::max( std::max( pt1(0,0), pt2(0,0) ), std::max( pt3(0,0), pt4(0,0) ) ) - x;
    height      =   std::max( std::max( pt1(1,0), pt2(1,0) ), std::max( pt3(1,0), pt4(1,0) ) ) - y;
}


template <typename  T>
void ImageRectangle<T>::overlap_inPlace(const ImageRectangle<T> &rect) {
/**
 * Ð´Ð»Ñ Ð½Ð°Ñ…Ð¾Ð¶Ð´ÐµÐ½Ð¸Ñ Ð¿ÐµÑ€ÐµÑÐµÑ‡ÐµÐ½Ð¸Ñ  Ð´Ð²ÑƒÑ… Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð² Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ÑÑ  Ð¼ÐµÑ‚Ð¾Ð´ Ð·Ð°Ð¼ÐµÑ‚Ð°ÑŽÑ‰ÐµÐ¹ Ð¿Ñ€ÑÐ¼Ð¾Ð¹ (sweep-line algorithm)
 *  Ð´Ð°Ð½Ð½Ð°Ñ Ð´Ð²ÑƒÐ¼ÐµÑ€Ð½Ð°Ñ Ð·Ð°Ð´Ð°Ñ‡Ð° Ñ€Ð°Ð·Ð±Ð¸Ð²Ð°ÐµÑ‚ÑÑ Ð½Ð° Ð´Ð²Ðµ Ð¾Ð´Ð½Ð¾Ð¼ÐµÑ€Ð½Ñ‹Ñ…. Ð¡Ð½Ð°Ñ‡Ð°Ð»Ð°  Ð½Ð°  Ð¾ÑÑŒ  Ð¥ Ð¿Ñ€Ð¾Ñ†Ð¸Ñ€ÑƒÑŽÑ‚ÑÑ  Ð½Ð°Ñ‡Ð°Ð»Ð¾ Ð¸ ÐºÐ¾Ð½ÐµÑ†  Ð´Ð²ÑƒÑ…  Ð¿Ñ€Ð¼ÑÐ¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð² Ð¿Ð¾ Ð¥ , Ð° Ð½Ð°
 *  Ð¾ÑÑŒ Y -- Ð½Ð°Ñ‡Ð°Ð»Ð¾ Ð¸ ÐºÐ¾Ð½ÐµÑ† Ð¿Ð¾ Y.  Ð”Ð°Ð»ÐµÐµ Ñ€ÐµÑˆÐ°ÐµÑ‚ÑÑ Ð¿Ñ€Ð¾ÑÑ‚Ð°Ñ Ð·Ð°Ð´Ð°Ñ‡Ð° Ð½Ð°Ñ…Ð¾Ð¶Ð´ÐµÐ½Ð¸Ñ Ð¿ÐµÑ€ÐµÑÐµÑ‡Ð²ÐµÐ½Ð¸Ñ Ð¾Ñ‚Ñ€ÐµÐ·ÐºÐ¾Ð² Ð½Ð° Ð¿Ñ€ÑÐ¼Ð¾Ð¹. ÐÐ°Ð¹Ð´ÐµÐ½Ð½Ñ‹Ðµ Ð¾Ñ‚Ñ€ÐµÐ·ÐºÐ¸ Ð¿ÐµÑ€ÐµÑÐµÑ‡ÐµÐ½Ð¸Ñ Ð¿Ð¾ X Ð¸ Y
 *  Ð¸ ÑÐ²Ð»ÑÑŽÑ‚ ÑÐ¾Ð±Ð¾Ð¹ Ð¸ÑÐºÐ¾Ð¼Ñ‹Ð¹ Ð¿Ñ€ÑÑ‡Ð¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸Ðº ( ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ð¹ ÐµÑÑ‚ÑŒ Ð¿ÐµÑ€ÐµÑÐµÑ‡ÐµÐ½Ð¸Ðµ Ð´Ð²ÑƒÑ… Ð¸ÑÑ…Ð¾Ð´Ð½Ñ‹Ñ… ). Ð”Ð°Ð½Ð½Ñ‹Ð¹ Ð¼ÐµÑ‚Ð¾Ð´Ð° ÐµÑÑ‚ÐµÑÑ‚Ð²ÐµÐ½Ð½Ð¾ Ð¾Ð±Ð¾Ñ‰Ð°ÐµÑ‚ÑÑ Ð½Ð° ÑÐ»ÑƒÑ‡Ð°Ð¹ k Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð².
 * */

    if  (  this->check_isEmpty() )   return;
    if  (  rect.check_isEmpty()  )   return;

    T segments_x_begin[2],          segments_x_end[2];
    T segments_y_begin[2],          segments_y_end[2];
    T segments_x_intersection[2],   segments_y_intersection[2];

    /** projecting horizontal side of the rectangles to  X axis */

    segments_x_begin[0]         =  this->x;
    segments_x_end[0]           =  this->x + this->width;

    segments_x_begin[1]         =  rect.x;
    segments_x_end[1]           =  rect.x+rect.width;

    /** projecting vertical side of the rectangles to  Y axis */

    segments_y_begin[0]         =  this->y;
    segments_y_end[0]           =  this->y + this->height;

    segments_y_begin[1]         =  rect.y;
    segments_y_end[1]           =  rect.y+rect.height;

    /** finding intersection of projected segments */

    segments_x_intersection[0]  =     std::max( segments_x_begin[0], segments_x_begin[1] );
    segments_x_intersection[1]  =     std::min( segments_x_end[0],   segments_x_end[1]   );


    segments_y_intersection[0]  =     std::max( segments_y_begin[0], segments_y_begin[1] );
    segments_y_intersection[1]  =     std::min( segments_y_end[0],   segments_y_end[1]   );


    /** check if rectangles have non empty intersection */

    if ( segments_x_intersection[1] < segments_x_intersection[0] || segments_y_intersection[1] < segments_y_intersection[0] ) {

        return;
    }

    /**do inPlace assignment*/

    this->x                     =   segments_x_intersection[0];
    this->width                 =   segments_x_intersection[1] - segments_x_intersection[0];

    this->y                     =   segments_y_intersection[0];
    this->height                =   segments_y_intersection[1] - segments_y_intersection[0];

//    this->is_empty              = check_isEmpty();
}




template <typename  T>
ImageRectangle<T>  ImageRectangle<T>::overlap(const ImageRectangle<T> &rect) {

    ImageRectangle<T> rect_return(x,y,width,height,border);

    rect_return.overlap_inPlace(rect);

    return rect_return;
}



template <typename  T>
void ImageRectangle<T>::circumscribe_inPlace(const ImageRectangle<T> &rect) {
/**
 * Ð´Ð»Ñ Ð½Ð°Ñ…Ð¾Ð¶Ð´ÐµÐ½Ð¸Ñ Ð¾Ð¿Ð¸ÑÐ°Ð½Ð½Ð¾Ð³Ð¾ Ð²Ð¾ÐºÑ€ÑƒÐ³ Ð´Ð²ÑƒÑ… Ð´Ð°Ð½Ð½Ñ‹Ñ…  Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð² Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ÑÑ  Ð¼ÐµÑ‚Ð¾Ð´ Ð·Ð°Ð¼ÐµÑ‚Ð°ÑŽÑ‰ÐµÐ¹ Ð¿Ñ€ÑÐ¼Ð¾Ð¹ (sweep-line algorithm).
 * Ð¡Ð»Ð¾Ð²Ð¾ "Ð¾Ð±ÑŠÐ´Ð¸Ð½ÐµÐ½Ð¸Ðµ" Ð½Ðµ Ð¸ÑÐ¿Ð¾Ð»ÑŒÐ·ÑƒÐµÑ‚ÑÑ, Ð¿Ð¾Ñ‚Ð¾Ð¼Ñƒ Ñ‡Ñ‚Ð¾ Ð¾Ð±ÑŠÐµÐ´Ð¸Ð½ÐµÐ½Ð¸Ðµ Ð´Ð²ÑƒÑ… Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð² Ð¼Ð¾Ð¶ÐµÑ‚ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð¼ Ð¸ Ð½Ðµ Ð±Ñ‹Ñ‚ÑŒ.
 *  Ð”Ð°Ð½Ð½Ð°Ñ Ð´Ð²ÑƒÐ¼ÐµÑ€Ð½Ð°Ñ Ð·Ð°Ð´Ð°Ñ‡Ð° Ñ€Ð°Ð·Ð±Ð¸Ð²Ð°ÐµÑ‚ÑÑ Ð½Ð° Ð´Ð²Ðµ Ð¾Ð´Ð½Ð¾Ð¼ÐµÑ€Ð½Ñ‹Ñ…. Ð¡Ð½Ð°Ñ‡Ð°Ð»Ð°  Ð½Ð°  Ð¾ÑÑŒ  Ð¥ Ð¿Ñ€Ð¾Ñ†Ð¸Ñ€ÑƒÑŽÑ‚ÑÑ  Ð½Ð°Ñ‡Ð°Ð»Ð¾ Ð¸ ÐºÐ¾Ð½ÐµÑ†  Ð´Ð²ÑƒÑ…  Ð¿Ñ€Ð¼ÑÐ¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð² Ð¿Ð¾ Ð¥ , Ð° Ð½Ð°
 *  Ð¾ÑÑŒ Y -- Ð½Ð°Ñ‡Ð°Ð»Ð¾ Ð¸ ÐºÐ¾Ð½ÐµÑ† Ð¿Ð¾ Y.  Ð”Ð°Ð»ÐµÐµ Ñ€ÐµÑˆÐ°ÐµÑ‚ÑÑ Ð¿Ñ€Ð¾ÑÑ‚Ð°Ñ Ð·Ð°Ð´Ð°Ñ‡Ð° Ð½Ð°Ñ…Ð¾Ð¶Ð´ÐµÐ½Ð¸Ñ Ð½Ð°Ð¸Ð¼ÐµÐ½ÑŒÑˆÐµÐ³Ð¾ Ð¾Ñ‚Ñ€ÐµÐ·ÐºÐ°, ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ð¹ ÑÐ¾Ð´ÐµÑ€Ð¶Ð¸Ñ‚ Ð´Ð²Ð° Ð¸ÑÑ…Ð¾Ð´Ð½Ñ‹Ñ… Ð¾Ñ‚Ñ€ÐµÐ·ÐºÐ° Ð½Ð° Ð¿Ñ€ÑÐ¼Ð¾Ð¹. ÐÐ°Ð¹Ð´ÐµÐ½Ð½Ñ‹Ðµ Ð¾Ñ‚Ñ€ÐµÐ·ÐºÐ¸  Ð¿Ð¾ X Ð¸ Y
 *  Ð¸ ÑÐ²Ð»ÑÑŽÑ‚ ÑÐ¾Ð±Ð¾Ð¹ Ð¸ÑÐºÐ¾Ð¼Ñ‹Ð¹ Ð¿Ñ€ÑÑ‡Ð¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸Ðº ( ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ð¹ ÐµÑÑ‚ÑŒ Ð¾Ð¿Ð¸ÑÐ°Ð½Ð½Ñ‹Ð¹ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸Ðº Ð²Ð¾ÐºÑ€ÑƒÐ³ Ð´Ð²ÑƒÑ… Ð¸ÑÑ…Ð¾Ð´Ð½Ñ‹Ñ… ). Ð”Ð°Ð½Ð½Ñ‹Ð¹ Ð¼ÐµÑ‚Ð¾Ð´ ÐµÑÑ‚ÐµÑÑ‚Ð²ÐµÐ½Ð½Ð¾ Ð¾Ð±Ð¾Ñ‰Ð°ÐµÑ‚ÑÑ Ð½Ð° ÑÐ»ÑƒÑ‡Ð°Ð¹ k Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð².
 * */

    if( rect.check_isEmpty() ) return;

    T segments_x_begin[2],          segments_x_end[2];
    T segments_y_begin[2],          segments_y_end[2];
    T segments_x_circumscribtion[2],   segments_y_circumscribtion[2];

    /** projecting horizontal side of the rectangles to  X axis */

    segments_x_begin[0]         =  this->x;
    segments_x_end[0]           =  this->x + this->width;

    segments_x_begin[1]         =  rect.x;
    segments_x_end[1]           =  rect.x+rect.width;

    /** projecting vertical side of the rectangles to  Y axis */

    segments_y_begin[0]         =  this->y;
    segments_y_end[0]           =  this->y + this->height;

    segments_y_begin[1]         =  rect.y;
    segments_y_end[1]           =  rect.y+rect.height;

    /** circumscribe projected segments */
/*
    segments_x_circumscribtion[0]  =     std::min( segments_x_begin[0], segments_x_begin[1] );
    segments_x_circumscribtion[1]  =     std::max( segments_x_end[0],   segments_x_end[1]   );


    segments_y_circumscribtion[0]  =     std::min( segments_y_begin[0], segments_y_begin[1] );
    segments_y_circumscribtion[1]  =     std::max( segments_y_end[0],   segments_y_end[1]   );
*/

    T new_x = std::min( x, rect.x);
    T new_y = std::min( y, rect.y);

    width     = std::max( x+ width,  rect.x + rect.width ) - new_x;
    height    = std::max( y+ height, rect.y + rect.height) - new_y;


    x = new_x;
    y = new_y;

    /**do inPlace assignment*/

/*
    this->x                     =   segments_x_circumscribtion[0];
    this->width                 =   segments_x_circumscribtion[1] - segments_x_circumscribtion[0];

    this->y                     =   segments_y_circumscribtion[0];
    this->height                =   segments_y_circumscribtion[1] - segments_y_circumscribtion[0];
*/



//    this->is_empty              = check_isEmpty();

}

template <typename  T>
ImageRectangle<T>  ImageRectangle<T>::maxSizeTo( const ImageRectangle<T> &rect ){


    return ImageRectangle<T>(x,y, std::max(width, rect.width), std::max(height, rect.height ));
}


template <typename  T>
ImageRectangle<T>  ImageRectangle<T>::minSizeTo( const ImageRectangle<T> &rect ){


    return ImageRectangle<T>(x,y, std::min(width, rect.width), std::min(height, rect.height ));
}


template <typename  T>
ImageRectangle<T>  ImageRectangle<T>::circumscribe(const ImageRectangle<T> &rect) {

    ImageRectangle<T> rect_return(x,y,width,height,border);

    rect_return.circumscribe_inPlace( rect );

    return rect_return;
}

template<typename T>
void ImageRectangle<T>::set(T x_, T y_, T width_, T height_, T border_) {


    if ( border_ > 0 )
        border  = border_;
    else
        border  = 0;

    set(x_,y_, height_, width_);

}

template<typename T>
void ImageRectangle<T>::set(T x_, T y_, T width_, T height_) {

    x = x_;
    y = y_;

    set_size(width_, height_);

}

template<typename T>
void ImageRectangle<T>::set_size(T width_, T height_) {


    if ( width_ > 0)
        width = width_;
    else {
        return;
    }


    if ( height_ > 0)
        height = height_;
    else{
        return;
    }
}

template<typename T>
void ImageRectangle<T>::set(T x_, T y_) {

    x   = x_;
    y   = y_;
}



/** get methods*/

template <typename  T>
void   ImageRectangle<T>::get( T &x_, T &y_, T &width_, T &height_, T &border_) const{

    get             ( x_, y_, width_, height_);
    get_border      ( border_);
}

template <typename  T>
void  ImageRectangle<T>::get(T &x_, T &y_, T &width_, T &height_) const {

   get          ( x_,     y_);
   get_size     ( width_, height_);
}

template <typename  T>
void  ImageRectangle<T>::get (  T& x_, T& y_) const {

    x_          =   x;
    y_          =   y;
}

template <typename  T>
void  ImageRectangle<T>::get_size (  T& width_, T& height_) const{

    width_      =   width;
    height_     =   height;
}

template <typename  T>
void   ImageRectangle<T>::get_border (  T& border_ ) const{

    border_     = border;
}


template <typename  T>
void ImageRectangle<T>::print() {

    if ( check_isEmpty()){
        std::cout << "rectangle is empty" << std::endl;
        return;
    }

    std::cout << "from [" << x << ", " << y <<  "]" << " with size " << width << " x " << height << std::endl;

}


template <typename  T>
cv::Point_<T> ImageRectangle<T>::get_ulOpenCV() const{

    ///  Ð¿Ð¾Ð»ÑƒÑ‡Ð¸Ñ‚ÑŒ Ð»ÐµÐ²Ñ‹Ð¹ Ð²ÐµÑ€Ñ…Ð½Ð¸Ð¹ ÑƒÐ³Ð¾Ð» Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° Ð¸Ð»Ð¸ "Ñ‚Ð¾Ñ‡ÐºÑƒ Ð¾Ñ‚ÑÑ‡Ñ‘Ñ‚a"
    cv::Point_<T> out( x,y );

    return out;
}


template <typename  T>
cv::Point_<T> ImageRectangle<T>::get_brOpenCV() const{

    ///  Ð¿Ð¾Ð»ÑƒÑ‡Ð¸Ñ‚ÑŒ Ð»ÐµÐ²Ñ‹Ð¹ Ð²ÐµÑ€Ñ…Ð½Ð¸Ð¹ ÑƒÐ³Ð¾Ð» Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° Ð¸Ð»Ð¸ "Ñ‚Ð¾Ñ‡ÐºÑƒ Ð¾Ñ‚ÑÑ‡Ñ‘Ñ‚a"
    cv::Point_<T> out( x+width,y+height );

    return out;
}

template <typename  T>
cv::Point_<T> ImageRectangle<T>::get_whOpenCV( )     const{

    cv::Point_<T> out( width,height );

    return out;

}


template <typename  T>
cv::Size_<T> ImageRectangle<T>::get_whSizeCV( )     const{

    cv::Size_<T> out( width, height);

    return  out;
}


template <typename  T>
Eigen::Matrix<T, 2, 1>  ImageRectangle<T>::get_ul() const{

    ///  Ð¿Ð¾Ð»ÑƒÑ‡Ð¸Ñ‚ÑŒ Ð»ÐµÐ²Ñ‹Ð¹ Ð²ÐµÑ€Ñ…Ð½Ð¸Ð¹ ÑƒÐ³Ð¾Ð» Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° Ð¸Ð»Ð¸ "Ñ‚Ð¾Ñ‡ÐºÑƒ Ð¾Ñ‚ÑÑ‡Ñ‘Ñ‚a"
    Eigen::Matrix<T, 2, 1> out;

    out(0,0)  = x;
    out(1,0)  =  y;

    return out;
}

template <typename  T>
Eigen::Matrix<T, 2, 1>  ImageRectangle<T>::get_ur() const{

    ///  Ð¿Ð¾Ð»ÑƒÑ‡Ð¸Ñ‚ÑŒ Ð»ÐµÐ²Ñ‹Ð¹ Ð²ÐµÑ€Ñ…Ð½Ð¸Ð¹ ÑƒÐ³Ð¾Ð» Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° Ð¸Ð»Ð¸ "Ñ‚Ð¾Ñ‡ÐºÑƒ Ð¾Ñ‚ÑÑ‡Ñ‘Ñ‚a"
    Eigen::Matrix<T, 2, 1> out;

    out(0,0)  = calc_maxX();
    out(1,0)  = y;

    return out;
}

template <typename  T>
Eigen::Matrix<T, 2, 1>  ImageRectangle<T>::get_br() const{

    ///  Ð¿Ð¾Ð»ÑƒÑ‡Ð¸Ñ‚ÑŒ Ð»ÐµÐ²Ñ‹Ð¹ Ð²ÐµÑ€Ñ…Ð½Ð¸Ð¹ ÑƒÐ³Ð¾Ð» Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° Ð¸Ð»Ð¸ "Ñ‚Ð¾Ñ‡ÐºÑƒ Ð¾Ñ‚ÑÑ‡Ñ‘Ñ‚a"

    Eigen::Matrix<T, 2, 1> out;

    out(0,0)  =  x + width;
    out(1,0)  =  y + height;

    return out;
}
template <typename  T>
Eigen::Matrix<T, 2, 1>  ImageRectangle<T>::get_bl() const{

    ///  Ð¿Ð¾Ð»ÑƒÑ‡Ð¸Ñ‚ÑŒ Ð»ÐµÐ²Ñ‹Ð¹ Ð²ÐµÑ€Ñ…Ð½Ð¸Ð¹ ÑƒÐ³Ð¾Ð» Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° Ð¸Ð»Ð¸ "Ñ‚Ð¾Ñ‡ÐºÑƒ Ð¾Ñ‚ÑÑ‡Ñ‘Ñ‚a"

    Eigen::Matrix<T, 2, 1> out;

    out(0,0)  =  x;
    out(1,0)  =  calc_maxY();

    return out;
}

/*

template <typename  T>
void ImageRectangle<T>::get_ul(Eigen::Matrix<T, 2, 1> &up_left_corner) const{

    ///  Ð¿Ð¾Ð»ÑƒÑ‡Ð¸Ñ‚ÑŒ Ð»ÐµÐ²Ñ‹Ð¹ Ð²ÐµÑ€Ñ…Ð½Ð¸Ð¹ ÑƒÐ³Ð¾Ð» Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° Ð¸Ð»Ð¸ "Ñ‚Ð¾Ñ‡ÐºÑƒ Ð¾Ñ‚ÑÑ‡Ñ‘Ñ‚a"

    up_left_corner(0,0)  =  x;
    up_left_corner(1,0)  =  y;

}

*/


template<class T>
void    ImageRectangle<T>::fitSizeTo  (  T  new_width, T new_height ){

    T center_x  =   x + 0.5*width ;
    T center_y  =   y + 0.5*height;

    x           =   center_x - 0.5*new_width;
    y           =   center_y - 0.5*new_height;

    width       =   new_width;
    height      =   new_height;

}


template<class T>
void ImageRectangle<T>::scaleSizeTo( T scale ){

    width *= scale;
    height*= scale;
}


template<class T>
void ImageRectangle<T>::fitSizeTo(const ImageRectangle<T> &rect) {

    /**
     *  Ð¿Ñ€Ð¸Ð²Ð¾Ð´Ð¸Ð¼ Ñ€Ð°Ð·Ð¼ÐµÑ€ ÑÑ‚Ð¾Ð³Ð¾ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° Ð² Ñ€Ð°Ð·Ð¼ÐµÑ€Ñƒ Ð´Ñ€ÑƒÐ³Ð¾Ð³Ð¾ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ° rect
     *  Ð”Ð°Ð½Ð½ÑƒÑŽ Ð¾Ð¿ÐµÑ€Ð°Ñ†Ð¸ÑŽ Ð¼Ð¾Ð¶Ð½Ð¾ Ð¼Ñ‹ÑÐ»Ð¸Ñ‚ÑŒ ÐºÐ°Ðº Ð½ÐµÑ€Ð°Ð²Ð½Ð¾Ð¼ÐµÑ€Ð½Ð¾Ðµ Ð¼Ð°ÑÑˆÑ‚Ð°Ð±Ð¸Ñ€Ð¾Ð²Ð°Ð½Ð¸Ðµ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸Ð° Ð¾Ñ‚Ð½Ð¾ÑÐ¸Ñ‚ÐµÐ»ÑŒÐ½Ð¾ ÐµÐ³Ð¾ Ñ†ÐµÐ½Ñ‚Ñ€Ð° (Ñ‚Ð°Ðº, Ñ‡Ñ‚Ð¾Ð±Ñ‹ Ñ€Ð°Ð·Ð¼ÐµÑ€Ñ‹ ÑÐ¾Ð²Ð¿Ð°Ð»Ð¸ Ñ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð¼ rect)
     *  Ð²ÐµÑ€Ñ…Ð½Ð¸Ð¹ Ð»ÐµÐ²Ñ‹Ð¹ ÑƒÐ³Ð¾Ð» rect Ð² Ð²Ñ‹Ñ‡Ð¸ÑÐ»ÐµÐ½Ð¸ÑÑ… Ð½Ðµ ÑƒÑ‡Ð°Ð²ÑÑ‚Ð²ÑƒÐµÑ‚
     * */

    T center_x  =   x + 0.5*width ;
    T center_y  =   y + 0.5*height;

    x           =   center_x - 0.5*rect.width;
    y           =   center_y - 0.5*rect.height;

    width       =   rect.width;
    height      =   rect.height;

}

template<class T>
bool ImageRectangle<T>::check_isEmpty() const {

    return !(width > 0 && height > 0);

}

template<class T>
void ImageRectangle<T>::shrinkBy(T factor) {

    if (  factor <= 0 || check_isEmpty() ) return;

    T  shrink_width  =  width - factor;
    T  shrink_height =  height- factor;

    if ( shrink_height <=0 || shrink_width <= 0 ) return;

    width  = shrink_width;
    height = shrink_height;

    x +=  0.5*factor;
    y +=  0.5*factor;


}

template<class T>
cv::Rect_<T> ImageRectangle<T>::get_OpencV() const {

    return cv::Rect_<T>(x,y,width, height);
}

template<class T>
void ImageRectangle<T>::set_ul2zero() {

    x = 0;
    y = 0;

}

template<class T>
Eigen::Matrix<T, 2, 4> ImageRectangle<T>::get_corners() const {

    Eigen::Matrix<T, 2, 4> out;

    out(0,0) = x;
    out(1,0) = y;

    out(0,1) = x+width;
    out(1,1) = y;

    out(0,2) = x+width;
    out(1,2) = y+height;

    out(0,3) = x;
    out(1,3) = y+height;


    return out;
}


template<class T>
void ImageRectangle<T>::get_regularGrid(Eigen::Vector2i divs, Eigen::Matrix<double, 2, Eigen::Dynamic> &out) const {


    Eigen::Matrix<T, 2,1>       wh              =   get_wh();

    wh(0) -= 1;    wh(1) -= 1;

    Eigen::Vector2d             div_inverse     =   (divs.cast<double>() - Eigen::Vector2d(1,1)).cwiseInverse();
    Eigen::Vector2d             step_size       =   Eigen::Vector2d(wh(0,0), wh(1,0)).cwiseProduct(  div_inverse );
    int                         col_idx         =   0;

    out.resize(2, divs(0)*divs(1));


    for (int idx2 = 0; idx2 < divs(1); ++idx2) {
        for (int idx1 = 0; idx1 < divs(0); ++idx1) {

            out.col(col_idx) = Eigen::Vector2d(idx1, idx2).cwiseProduct( step_size)  + Eigen::Vector2d(x,y);
            col_idx++;
        }
    }

}


template<class T>
Eigen::Matrix<T, 2, 1> ImageRectangle<T>::get_wh() const {

    return Eigen::Matrix<T, 2, 1>(width, height);
}

template<class T>
void ImageRectangle<T>::shiftBy(T x, T y) {

    this->x += x;
    this->y += y;
}


template<class T>
void ImageRectangle<T>::shiftBy ( Eigen::Matrix<T, 2, 1> v){

    this->x += v(0);
    this->y += v(1);
}


template<class T>
void ImageRectangle<T>::cutOutsidePoints(const Eigen::Matrix2Xd &points_in, Eigen::Matrix2Xd &points_out) {

    /// cut (or just delete) points, which are outside rectangle bounds

    int                 num_points          = points_in.cols();
    int                 num_points_new      (0);
    std::vector<bool>   in_points_labels    (num_points);

    std::fill(in_points_labels.begin(), in_points_labels.end(), false);


    for (int idx = 0; idx < num_points; ++idx) {

        Eigen::Vector2d pt = points_in.col(idx);

        if ( pt(0)>=x  &&  pt(1)>=y  &&  pt(0)<=x+width  &&  pt(1)<=y+height ) {

            in_points_labels[idx] = true;
            num_points_new++;
        }
    }


    points_out.conservativeResize(Eigen::NoChange, num_points_new);
    int idx_new(0);


    for (int idx = 0; idx < num_points; ++idx) {

        if ( in_points_labels[idx] ) {

            points_out.col(idx_new) = points_in.col(idx);
            idx_new++;
        }
    }
}

template<class T>
void ImageRectangle<T>::set_ul(const Eigen::Matrix<T, 2, 1> &ul) {

    x = ul(0);
    y = ul(1);
}


template<class T>
void ImageRectangle<T>::pushInside(ImageRectangle<T> rect, Eigen::Matrix<T, 2, 1> *shift) {

    /// Ð¿Ð¾Ð¼ÐµÑ‰Ð°ÐµÐ¼ Ð´Ð°Ð½Ñ‹Ð½Ð¹ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸Ðº (this) Ð²Ð½ÑƒÑ‚Ñ€ÑŒ Ð´Ñ€ÑƒÐ³Ð¾Ð³Ð¾ (rect)

    if (  rect.check_hasInside(*this) ){

        if ( shift != nullptr)

            shift->setZero();
            return;
    }
    if ( width > rect.width || height > rect.height){ std::cerr << "can't push inside, rectangle's width or height exceeds current rectangle dimensions \n"; return; }


    Eigen::Matrix<T, 2, 4>                  corners = get_corners();
    std::vector<double>                     dist_from_corners_to_rect;
    std::vector<Eigen::Matrix<T, 2, 1>>     vec_from_corners_to_rect;


    for (int idx = 0; idx < 4; ++idx) {

        if ( !rect.check_hasInside<T>(corners.col(idx)) ) {

            Eigen::Matrix<T, 2, 1>  closest_vec     =  rect.closestVectorFromPoint(corners.col(idx));
            Eigen::Vector2d         closest_vec_d   =  closest_vec.template cast<double>();

            vec_from_corners_to_rect.push_back   ( closest_vec );
            dist_from_corners_to_rect.push_back  ( closest_vec_d.norm() );
        }
    }


    typename std::vector<double>::iterator      max_element_it;
    int                                         max_element_idx;

    max_element_it      =   std::max_element( dist_from_corners_to_rect.begin(), dist_from_corners_to_rect.end() );
    max_element_idx     =   std::distance( dist_from_corners_to_rect.begin(), max_element_it);


    shiftBy( -vec_from_corners_to_rect[max_element_idx] );

    if ( shift != nullptr){
        *shift = -vec_from_corners_to_rect[max_element_idx];
    }
}


template<class T>
template<typename T2>
bool ImageRectangle<T>::check_hasInside(Eigen::Matrix<T2, 2, 1> pt) const{


    return (x <= pt(0) && pt(0) <= x + width) && (y <= pt(1) && pt(1) <= y + height);
}

template<class T>
bool ImageRectangle<T>::check_hasInside(ImageRectangle<T> rect) const {

    if (check_hasInside(rect.get_ul())  && check_hasInside(rect.get_br()))
        return  true;
    else
        return false;
}


template<class T>
Eigen::Matrix<T, 2, 1> ImageRectangle<T>::closestVectorFromPoint(Eigen::Matrix<T, 2, 1> pt) {

    /**
     * Ð½Ð°Ñ…Ð¾Ð´Ð¸Ñ‚ÑÑ Ð²ÐµÐºÑ‚Ð¾Ñ€ Ñ Ð½Ð°Ñ‡Ð°Ð»Ð¾Ð¼ Ð² Ð´Ð°Ð½Ð½Ð¾Ð¹ Ñ‚Ð¾Ñ‡ÐºÐ¸ Ð¸ ÐºÐ¾Ð½Ñ†Ð¾Ð¼, Ð»ÐµÐ¶Ð°Ñ‰ÐµÐ¼ Ð½Ð° Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐµ Ñ Ð¼Ð¸Ð½Ð¸Ð¼Ð°Ð»ÑŒÐ½Ð¾Ð¹ Ð´Ð»Ð¸Ð½Ð½Ð¾Ð¹
     * */


    if ( pt(0) >= x && pt(0) <= x+width ){

        Eigen::Matrix<T, 2, 1>  vec;

        if ( pt(1) < y)
            vec << 0,   pt(1)-y;

        if (pt(1)  >y+height)
            vec << 0,   pt(1)-y-height;


        return vec;
    }

    if ( pt(1) >= y && pt(1) <= y+height){

        Eigen::Matrix<T, 2, 1>  vec;

        if ( pt(0) < x )
            vec << pt(0)-x, 0 ;

        if ( pt(0) > x+width )
            vec << pt(0)-x-width, 0 ;

        return vec;
    }


    Eigen::Matrix<T, 2, 4>              corners   = get_corners();
    std::vector<T>                      dist_to_corners(4);

    for (int idx = 0; idx < 4; ++idx) {
        dist_to_corners[idx] = ( pt - corners.col(idx) ).norm();
    }

    typename std::vector<T>::iterator   min_element_it;
    int                                 min_element_idx;

    min_element_it      =   std::min_element( dist_to_corners.begin(), dist_to_corners.end() );
    min_element_idx     =   std::distance( dist_to_corners.begin(), min_element_it);

    return  pt - corners.col(min_element_idx);
}

template<class T>
void ImageRectangle<T>::slice(ImageRectangle<T> *pRectSlice, ImageRectangle<T> *pRectRemainder, T amount, RectangleEdge edge) {


    switch (edge){
        /// =================================================== X EDGES
        case RectangleEdge::minXEdge:

            if ( amount <= 0){

                if ( pRectSlice     != nullptr)      *pRectSlice         = cv::Rect(0,0,0,0);
                if ( pRectRemainder != nullptr)      *pRectRemainder     = *this;
            }
            if ( amount > 0 && amount < this->width ){

                if ( pRectSlice     != nullptr)      *pRectSlice         = cv::Rect( x,        y, amount,       height );
                if ( pRectRemainder != nullptr)      *pRectRemainder     = cv::Rect( x+amount, y, width-amount, height );
            }
            if ( amount >= this->width ){

                if ( pRectSlice     != nullptr)      *pRectSlice         = *this;
                if ( pRectRemainder != nullptr)      *pRectRemainder     = cv::Rect(0,0,0,0);
            }
            break;

        case RectangleEdge::maxXEdge:

            if ( amount <= 0){

                if ( pRectSlice     != nullptr)      *pRectSlice         = cv::Rect(0,0,0,0);
                if ( pRectRemainder != nullptr)      *pRectRemainder     = *this;
            }
            if ( amount > 0 && amount < this->width ){

                if ( pRectSlice     != nullptr)      *pRectSlice         = cv::Rect( x,               y,  width-amount, height );
                if ( pRectRemainder != nullptr)      *pRectRemainder     = cv::Rect( x+width-amount,  y,  amount,       height );
            }
            if ( amount >= this->width ){

                if ( pRectSlice     != nullptr)      *pRectSlice          = cv::Rect(0,0,0,0);
                if ( pRectRemainder != nullptr)      *pRectRemainder      = *this;
            }
            break;

            /// ================================================== Y EDGES
        case RectangleEdge::minYEdge:

            if ( amount <= 0){

                if ( pRectSlice     != nullptr)      *pRectSlice         = cv::Rect(0,0,0,0);
                if ( pRectRemainder != nullptr)      *pRectRemainder     = *this;
            }
            if ( amount > 0 && amount < this->height ){

                if ( pRectSlice     != nullptr)      *pRectSlice         = cv::Rect( x, y,        width, amount        );
                if ( pRectRemainder != nullptr)      *pRectRemainder     = cv::Rect( x, y+amount, width, height-amount );
            }
            if ( amount >= this->height ){

                if ( pRectSlice     != nullptr)      *pRectSlice         = *this;
                if ( pRectRemainder != nullptr)      *pRectRemainder     = cv::Rect(0,0,0,0);
            }
            break;

        case RectangleEdge::maxYEdge:

            if ( amount <= 0){

                if ( pRectSlice     != nullptr)      *pRectSlice          = *this;
                if ( pRectRemainder != nullptr)      *pRectRemainder      = cv::Rect(0,0,0,0);
            }
            if ( amount > 0 && amount < this->height ){

                if ( pRectSlice     != nullptr)      *pRectSlice          = cv::Rect( x, y,        width, amount        );
                if ( pRectRemainder != nullptr)      *pRectRemainder      = cv::Rect( x, y+amount, width, height-amount );
            }
            if ( amount >= this->height ){

                if ( pRectSlice     != nullptr)      *pRectSlice          = cv::Rect(0,0,0,0);
                if ( pRectRemainder != nullptr)      *pRectRemainder      = *this;
            }
            break;
    }

}

template<class T>
std::vector<ImageRectangle<T>> ImageRectangle<T>::subtract(const ImageRectangle<T> &rect) {

    /**
    rectangle1 \setminus rectangle2
    -------------------------
    |      rectangle 1      |
    |                       |
    |     -------------     |
    |     |rectangle 2|     |
    |     -------------     |
    |                       |
    |                       |
    -------------------------

    If you subtract rectangle 2 from rectangle 1, you will get an area with a hole. This area can be decomposed into 4 rectangles
    -------------------------
    |          A            |
    |                       |
    |-----------------------|
    |  B  |   hole    |  C  |
    |-----------------------|
    |                       |
    |          D            |
    -------------------------
*/



    if (this->calc_area() == 0 ) {
        return std::vector<ImageRectangle<T>>();
    }

    ImageRectangle<T> intersectedRect = this->overlap(rect); //rect1 | rect2;

    /// No intersection
    if (intersectedRect.calc_area() ==0 ) {

        std::vector<cv::Rect>  rect(1);
        rect[0] = *this;
        return rect;
    }


    std::vector<cv::Rect>   results;
    cv::Rect                remainder, subtractedArea;

    subtractedArea = rectBetween( *this, intersectedRect, &remainder, RectangleEdge::maxYEdge);
    if ( subtractedArea.area() != 0 ) {
        results.push_back(subtractedArea);
    }

    subtractedArea = rectBetween(remainder, intersectedRect, &remainder, RectangleEdge::minYEdge);
    if (subtractedArea.area() != 0 ) {
        results.push_back(subtractedArea);
    }

    subtractedArea = rectBetween(remainder, intersectedRect, &remainder, RectangleEdge::maxXEdge);
    if (subtractedArea.area() != 0 ) {
        results.push_back(subtractedArea);
    }

    subtractedArea = rectBetween(remainder, intersectedRect, &remainder, RectangleEdge::minXEdge);
    if (subtractedArea.area() != 0 ) {
        results.push_back(subtractedArea);
    }

    return results;
}

template<class T>
ImageRectangle<T> ImageRectangle<T>::rectBetween(const ImageRectangle<T> & rect1, const ImageRectangle<T> & rect2, ImageRectangle<T> *remainder, RectangleEdge edge) {

    /// returns the area between rect1 and rect2 along the edge

/*    cv::Rect intersectedRect = rect1 | rect2;

    if ( intersectedRect.calc_area() == 0) {
        return cv::Rect();
    }

    cv::Rect    rect3;
    float       chop_amount = 0;

    switch (edge) {

        case CGRectEdge::maxYEdge:
            chop_amount = rect1.height - (intersectedRect.y - rect1.y);
            if (chop_amount > rect1.height) { chop_amount = rect1.height; }
            break;

        case CGRectEdge::minYEdge:
            chop_amount = rect1.height - (CvRectGetMaxY(rect1) - CvRectGetMaxY(intersectedRect));
            if (chop_amount > rect1.height) { chop_amount = rect1.height; }
            break;

        case CGRectEdge::maxXEdge:
            chop_amount = rect1.width - (intersectedRect.x - rect1.x);
            if (chop_amount > rect1.width) { chop_amount = rect1.width; }
            break;

        case CGRectEdge::minXEdge:
            chop_amount = rect1.width - (CvRectGetMaxX(rect1) - CvRectGetMaxX(intersectedRect));
            if (chop_amount > rect1.width) { chop_amount = rect1.width; }
            break;

        default:
            break;
    }

    CvRectDivide(rect1, remainder, &rect3, chop_amount, edge);

    return rect3;*/

    return ImageRectangle();
}

template<class T>
T ImageRectangle<T>::calc_area() const {

    return width*height;
}

template<class T>
T ImageRectangle<T>::calc_maxX() const {

    return x+width;
}

template<class T>
T ImageRectangle<T>::calc_maxY()  const{

    return y+height;
}

template<class T>
Segment ImageRectangle<T>::get_sideSegment(ImageRectangle::RectangleEdge edge) {

    switch  (edge) {
        case RectangleEdge::minXEdge:
            return Segment(get_ul(), get_bl());
        case RectangleEdge::maxXEdge:
            return Segment(get_ur(), get_br());

        case RectangleEdge::minYEdge:
            return Segment(get_ul(), get_ur());
        case RectangleEdge::maxYEdge:
            return Segment(get_bl(), get_br());
    }

    return Segment();
}

template<class T>
std::vector<Segment> ImageRectangle<T>::get_sidesList() const {

    std::vector<Segment>   segments(4);

    ImageRectangleD rect = *this;

    segments[0] = Segment( rect.get_ul(), rect.get_ur() );
    segments[1] = Segment( rect.get_ur(), rect.get_br() );
    segments[2] = Segment( rect.get_br(), rect.get_bl() );
    segments[3] = Segment( rect.get_bl(), rect.get_ul() );

    return segments;
}

template<class T>
std::vector<ImageRectangle<T>> ImageRectangle<T>::subdivide(int level) {

    /**
     * Ð²Ð¾Ð·Ð²Ñ€Ð°Ñ‰Ð°ÐµÑ‚ level Ð² ÐºÐ²Ð°Ð´Ñ€Ð°Ñ‚Ðµ Ð¿Ñ€ÑÐ¼Ð¾ÑƒÐ³Ð¾Ð»ÑŒÐ½Ð¸ÐºÐ¾Ð², ÐºÐ¾Ñ‚Ð¾Ñ€Ñ‹Ðµ Ñ€Ð°Ð·Ð±Ð¸Ð²Ð°ÑŽÑ‚ Ð¸ÑÑ…Ð¾Ð´Ð½Ñ‹Ð¹
     * */

    std::vector<ImageRectangle<T>>  rects;
    int                             total_num_rects     =   std::pow(2, level);
    T                               width               =   this->width  /  level;
    T                               height              =   this->height /  level;


    for (int idx1=0; idx1 < level; ++idx1){
        for (int idx2=0; idx2 < level; ++idx2){

            ImageRectangle<T> rect(idx1*width, idx2+height, width, height);

            rects.push_back(rect);
        }
    }

    return rects;
}

template<class T>
Eigen::Matrix<T, 2, 1> ImageRectangle<T>::get_center() const {

    return Eigen::Matrix<T, 2, 1>(x+0.5 * width, y+0.5 * height);
}


#endif // IMAGERECTANGLE