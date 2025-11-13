#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

class SegmentationNode : public rclcpp::Node
{
public:
    SegmentationNode() : Node("segmentation_node")
    {
        RCLCPP_INFO(this->get_logger(), " SegmentationNode initialized (Ground removal + Object segmentation)");

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/filtered_points", 10,
            std::bind(&SegmentationNode::pointCloudCallback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/segmented_points", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty cloud");
            return;
        }

        // ------------------- STEP 1: PassThrough Filter (remove points below sensor / ground noise) -------------------
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-1.5, 3.0); // adjust for your LiDAR mounting height
        pass.filter(*cloud);

        // ------------------- STEP 2: Plane Segmentation (Ground removal using RANSAC) -------------------
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.15); // adjust based on road roughness (0.1–0.25 typical)
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty())
        {
            RCLCPP_WARN(this->get_logger(), " No ground plane detected!");
            return;
        }

        // ------------------- STEP 3: Extract Non-Ground Points (keep only objects) -------------------
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true); // keep everything except ground
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_objects(new pcl::PointCloud<pcl::PointXYZ>());
        extract.filter(*cloud_objects);

        if (cloud_objects->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No objects detected after ground removal!");
            return;
        }

        // ------------------- STEP 4: (Optional) Cluster pre-segmentation -------------------
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud_objects);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.5);
        ec.setMinClusterSize(30);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_objects);
        ec.extract(cluster_indices);

        pcl::PointCloud<pcl::PointXYZ>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (auto &indices : cluster_indices)
        {
            for (auto &idx : indices.indices)
                segmented_cloud->points.push_back(cloud_objects->points[idx]);
        }

        segmented_cloud->width = segmented_cloud->points.size();
        segmented_cloud->height = 1;
        segmented_cloud->is_dense = true;

        // ------------------- STEP 5: Publish Non-Ground (Segmented) Cloud -------------------
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*segmented_cloud, output);
        output.header = msg->header;
        publisher_->publish(output);

        RCLCPP_INFO(this->get_logger(), " Published segmented (non-ground) cloud with %zu points", segmented_cloud->points.size());
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SegmentationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
