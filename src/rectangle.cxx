// encapsulated #include <vulkan/vulkan.h>
#include <ostream>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <array>
#include <map>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <optional>
#include <set>

namespace
{

//------------------------------------------------------------
// Debug stuff
//-------------------------------------------------------------

// #define NDEBUG // uncomment to disable debug

#ifndef NDEBUG
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

bool checkValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> avaliableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, avaliableLayers.data());

    for (const char* layerName : validationLayers)
        if (std::find_if(avaliableLayers.begin(), avaliableLayers.end(),
                [layerName](const VkLayerProperties& layerProperty)
                {
                    return std::strcmp(layerName, layerProperty.layerName);
                })
                == avaliableLayers.end())
            return false;

    return true;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func)
        return func(instance, pCreateInfo, pAllocator, pMessenger);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func)
        return func(instance, debugMessenger, pAllocator);
}
#endif

const std::vector<const char*> deviceExtensions {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

bool checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> avaliableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, avaliableExtensions.data());

    for (const char* extensionName : deviceExtensions)
        if (avaliableExtensions.end() == std::find_if(avaliableExtensions.begin(), avaliableExtensions.end(),
                [extensionName](const VkExtensionProperties& extensionProperty)
                {
                    return std::strcmp(extensionName, extensionProperty.extensionName);
                }))
            return false;

    return true;
}


struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDesctiption()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

struct UniformBufferObject
{
    glm::mat4 mvp;
};

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const char* TITLE = "Triangle";
const uint32_t MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<Vertex> vertices = {
    {{0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}},
    {{- 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, - 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{- 0.5f, - 0.5f}, {0.0f, 0.0f, 1.0f}}
};

const std::vector<uint16_t> indices {
    0, 1, 2, 2, 1, 3
};
}

class HelloTriangleApplication
{
    GLFWwindow* window;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;

    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkDescriptorPool descriptorPool;
    VkCommandPool commandPool;

    uint32_t currentFrame = 0;
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    UniformBufferObject ubo;

    uint32_t maxFramesInFlight = MAX_FRAMES_IN_FLIGHT;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvaliableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    std::vector<VkDescriptorSet> descriptorSets;


#ifndef NDEBUG
    VkDebugUtilsMessengerEXT debugMessenger;

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
    {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.pNext = nullptr;
        createInfo.flags = 0;
        createInfo.messageSeverity =
            //VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr;
    }

    void setupDebugMessenger()
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
            throw std::runtime_error("Failed to set up debug messenger!");
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData)
    {
        std::cerr << pCallbackData->pMessage << std::endl;

        return VK_FALSE;
    }
#endif


public:
    bool framebufferResized = false;

    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }


private:
//------------------------------------------------------------
// Run functions
//------------------------------------------------------------
    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        HelloTriangleApplication* app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void initVulkan()
    {
        createInstance();
#ifndef NDEBUG
        setupDebugMessenger();
#endif
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        std::cerr << "ok\n";
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void cleanup()
    {
        cleanupSwapchain();

        for (VkBuffer& uniformBuffer : uniformBuffers)
            vkDestroyBuffer(device, uniformBuffer, nullptr);
        for (VkDeviceMemory& uniformBufferMemory : uniformBuffersMemory)
            vkFreeMemory(device, uniformBufferMemory, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        for (VkFence& inFlightFence : inFlightFences)
            vkDestroyFence(device, inFlightFence, nullptr);
        for (VkSemaphore& renderFinishedSemaphore : renderFinishedSemaphores)
            vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
        for (VkSemaphore& imageAvaliableSemaphore : imageAvaliableSemaphores)
            vkDestroySemaphore(device, imageAvaliableSemaphore, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
#ifndef NDEBUG
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
#endif
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void cleanupSwapchain()
    {
        for (VkFramebuffer& framebuffer : swapchainFramebuffers)
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        for (VkImageView& imageView : swapchainImageViews)
            vkDestroyImageView(device, imageView, nullptr);
        vkDestroySwapchainKHR(device, swapchain, nullptr);
    }

//------------------------------------------------------------
// GLFW dependent functions
//------------------------------------------------------------
    void createSurface()
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to create window surface!");
#ifndef NDEBUG
        std::cerr << "Created window surface " << surface << std::endl;
#endif
    }

    std::vector<const char*> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

#ifndef NDEBUG
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

        return extensions;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max() &&
            capabilities.currentExtent.height != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width,
                capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height,
                capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }

    void recreateSwapchain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        cleanupSwapchain();

        createSwapchain();
        createImageViews();
        createFramebuffers();
    }

//------------------------------------------------------------
// Init Vulkan functions
//------------------------------------------------------------
    void createInstance()
    {
#ifndef NDEBUG
        if (!checkValidationLayerSupport())
            throw std::runtime_error("Validation layers requested, but not avaliable!");
#endif
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pNext = nullptr;
        appInfo.pApplicationName = TITLE;
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

#ifndef NDEBUG
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        populateDebugMessengerCreateInfo(debugCreateInfo);
#endif
        std::vector<const char*> extensions = getRequiredExtensions();
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
#ifndef NDEBUG
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
#else
        createInfo.pNext = nullptr;
#endif
        createInfo.flags = 0;
        createInfo.pApplicationInfo = &appInfo;
#ifndef NDEBUG
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
#else
        createInfo.enabledLayerCount = 0;
        createInfo.ppEnabledLayerNames = nullptr;
#endif
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
            throw std::runtime_error("Failed to create instance!");
#ifndef NDEBUG
        std::cerr << "Created instance " << instance << std::endl;
#endif
    }

//------------------------------------------------------------
// Gpu check util functions begin
//------------------------------------------------------------
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
        static std::map<VkPhysicalDevice, QueueFamilyIndices> indicesMap;

        if (indicesMap.find(device) != indicesMap.end())
            return indicesMap[device];

        QueueFamilyIndices familyIndices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        uint32_t queueFamilyIndex = 0;
        for(const VkQueueFamilyProperties& queueFamily : queueFamilies)
        {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, presentSupport, surface, &presentSupport);

            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                familyIndices.graphicsFamily = queueFamilyIndex;

            if (presentSupport)
                familyIndices.presentFamily = queueFamilyIndex;

            if (familyIndices.isComplete())
                break;

            ++queueFamilyIndex;
        }

        indicesMap[device] = familyIndices;
        return familyIndices;
    }

    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device)
    {
        SwapchainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
        if (formatCount)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
        if (presentModeCount)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamilyIndices familyIndices = findQueueFamilies(device);

        bool extensionSupported = checkDeviceExtensionSupport(device);
        bool swapchainAdequate = false;
        if (extensionSupported)
        {
            SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device);
            swapchainAdequate = !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
        }

        return familyIndices.isComplete() && extensionSupported && swapchainAdequate;
    }
//------------------------------------------------------------
// Gpu check util functions end
//------------------------------------------------------------

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0)
            throw std::runtime_error("Failed to find GPUs with vulkan support!");

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        if (devices.end() == std::find_if(devices.begin(), devices.end(),
                [this](const VkPhysicalDevice& device)
                {
                    if (isDeviceSuitable(device))
                    {
                        physicalDevice = device;
                        return true;
                    }
                    return false;
                }))
            throw std::runtime_error("Failed to find a suitable gpu!");
#ifndef NDEBUG
        std::cerr << "Picked physical device " << physicalDevice << std::endl;
#endif
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices familyIndices = findQueueFamilies(physicalDevice);

        std::set<uint32_t> uniqueQueueFamilies = {familyIndices.graphicsFamily.value(), familyIndices.presentFamily.value()};
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos(uniqueQueueFamilies.size());

        float queuePriority = 1.0f;
        std::transform(uniqueQueueFamilies.begin(), uniqueQueueFamilies.end(), queueCreateInfos.begin(),
            [&queuePriority](uint32_t queueFamily)
            {
                VkDeviceQueueCreateInfo queueCreateInfo{};
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.pNext = nullptr;
                queueCreateInfo.flags = 0;
                queueCreateInfo.queueFamilyIndex = queueFamily;
                queueCreateInfo.queueCount = 1;
                queueCreateInfo.pQueuePriorities = &queuePriority;

                return queueCreateInfo;
            });

        VkPhysicalDeviceFeatures deviceFeatures{};
        vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = nullptr;
        createInfo.flags = 0;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
#ifndef NDEBUG
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
#else
        createInfo.enabledLayerCount = 0;
        createInfo.ppEnabledLayerNames = nullptr;
#endif
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        createInfo.pEnabledFeatures = &deviceFeatures;


        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device!");
#ifndef NDEBUG
        std::cerr << "Created device " << device << std::endl;
#endif

        vkGetDeviceQueue(device, familyIndices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, familyIndices.presentFamily.value(), 0, &presentQueue);
    }

//------------------------------------------------------------
// Swap chain util functions begin
//------------------------------------------------------------
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& avaliableFormats)
    {
        std::vector<VkSurfaceFormatKHR>::const_iterator itFoundFormat = std::find_if(avaliableFormats.begin(), avaliableFormats.end(),
            [](const VkSurfaceFormatKHR& avaliableFormat)
            {
                return avaliableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
                       avaliableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            });
        return itFoundFormat == avaliableFormats.end() ? avaliableFormats.front() : *itFoundFormat;
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& avaliablePresentModes)
    {
        return avaliablePresentModes.end() == std::find(avaliablePresentModes.begin(), avaliablePresentModes.end(),
                VK_PRESENT_MODE_MAILBOX_KHR) ? VK_PRESENT_MODE_FIFO_KHR : VK_PRESENT_MODE_MAILBOX_KHR;
    }
//------------------------------------------------------------
// Swap chain util functions end
//------------------------------------------------------------

    void createSwapchain()
    {
        SwapchainSupportDetails swapchainSupport = querySwapchainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapchainSupport.capabilities);

        uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;
        if (swapchainSupport.capabilities.maxImageCount)
            imageCount = std::clamp(imageCount,
                    swapchainSupport.capabilities.minImageCount, swapchainSupport.capabilities.maxImageCount);

        QueueFamilyIndices familyIndices = findQueueFamilies(physicalDevice);
        std::vector<uint32_t> queueFamilyIndices = {familyIndices.graphicsFamily.value(), familyIndices.presentFamily.value()};

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.pNext = nullptr;
        createInfo.flags = 0;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (familyIndices.graphicsFamily != familyIndices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = queueFamilyIndices.size();
            createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0;
            createInfo.pQueueFamilyIndices = nullptr;
        }
        createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS)
            throw std::runtime_error("Failed to create swapchain!");
#ifndef NDEBUG
        std::cerr << "Created swapchain " << swapchain << std::endl;
#endif

        swapchainImageFormat = surfaceFormat.format;
        swapchainExtent = extent;

        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
    }

    void createImageViews()
    {
        swapchainImageViews.resize(swapchainImages.size());
        std::transform(swapchainImages.begin(), swapchainImages.end(), swapchainImageViews.begin(),
            [this](const VkImage& swapchainImage)
            {
                VkImageViewCreateInfo createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                createInfo.pNext = nullptr;
                createInfo.flags = 0;
                createInfo.image = swapchainImage;
                createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                createInfo.format = swapchainImageFormat;
                createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;

                VkImageView imageView;
                if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS)
                    throw std::runtime_error("Faled to create image view!");

                return imageView;
            });
#ifndef NDEBUG
        std::cerr << "Created image views count " << swapchainImageViews.size() << std::endl;
#endif
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.flags = 0;
        colorAttachment.format = swapchainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentReference{};
        colorAttachmentReference.attachment = 0;
        colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.flags = 0;
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.inputAttachmentCount = 0;
        subpass.pInputAttachments = nullptr;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentReference;
        subpass.pResolveAttachments = nullptr;
        subpass.pDepthStencilAttachment = nullptr;
        subpass.preserveAttachmentCount = 0;
        subpass.pPreserveAttachments = nullptr;

        VkSubpassDependency dependency;
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependency.dependencyFlags = 0;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.pNext = nullptr;
        renderPassInfo.flags = 0;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
            throw std::runtime_error("Failed to create render pass!");
#ifndef NDEBUG
        std::cerr << "Created render pass " << renderPass << std::endl;
#endif
    }
    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding mvpLayoutBinding{};
        mvpLayoutBinding.binding = 0;
        mvpLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        mvpLayoutBinding.descriptorCount = 1;
        mvpLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        mvpLayoutBinding.pImmutableSamplers = nullptr;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.pNext = nullptr;
        layoutInfo.flags = 0;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &mvpLayoutBinding;

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor set layout!");
#ifndef NDEBUG
        std::cerr << "Created descriptor set layout" << descriptorSetLayout << std::endl;
#endif
    }

//------------------------------------------------------------
// Shader load util functions begin
//------------------------------------------------------------
    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("Failed to open file " + filename);

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pNext = nullptr;
        createInfo.flags = 0;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
            throw std::runtime_error("Failed to create shader moudle!");

        return shaderModule;
    }
//------------------------------------------------------------
// Shader load util functions end
//------------------------------------------------------------

    void createGraphicsPipeline()
    {
        const std::vector<std::pair<std::string, VkShaderStageFlagBits>> shaderList = {
            {"vert.spv", VK_SHADER_STAGE_VERTEX_BIT},
            {"tesc.spv", VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT},
            {"tese.spv", VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT},
            {"geom.spv", VK_SHADER_STAGE_GEOMETRY_BIT},
            {"frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT},
            {"comp.spv", VK_SHADER_STAGE_COMPUTE_BIT}
        };

        std::vector<std::filesystem::path> foundShaders;
        std::vector<VkShaderStageFlagBits> shaderStages;
        for (const std::pair<std::string, VkShaderStageFlagBits>& shaderParam : shaderList)
        {
            if (std::filesystem::exists(shaderParam.first))
            {
                foundShaders.push_back(shaderParam.first);
                shaderStages.push_back(shaderParam.second);
#ifndef NDEBUG
                std::cerr << "Found shader " << shaderParam.first << std::endl;
#endif
            }
        }

        std::vector<std::vector<char>> shaderCodes(foundShaders.size());
        std::transform(foundShaders.begin(), foundShaders.end(), shaderCodes.begin(),
            [](const std::string& filename)
            {
                return readFile(filename);
            });

        std::vector<VkShaderModule> shaderModules(foundShaders.size());
        std::transform(shaderCodes.begin(), shaderCodes.end(), shaderModules.begin(),
            [this](const std::vector<char>& code)
            {
                return createShaderModule(code);
            });

        std::vector<VkPipelineShaderStageCreateInfo> shaderInfos(foundShaders.size());
        std::transform(shaderModules.begin(), shaderModules.end(), shaderStages.begin(), shaderInfos.begin(),
            [](const VkShaderModule& shaderModule, const VkShaderStageFlagBits& stage)
            {
                VkPipelineShaderStageCreateInfo createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                createInfo.pNext = nullptr;
                createInfo.flags = 0;
                createInfo.stage = stage;
                createInfo.module = shaderModule;
                createInfo.pName = "main";

                return createInfo;
            });

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkVertexInputBindingDescription bindingDescription = Vertex::getBindingDesctiption();
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.pNext = nullptr;
        vertexInputInfo.flags = 0;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.pNext = nullptr;
        inputAssembly.flags = 0;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.pNext = nullptr;
        dynamicState.flags = 0;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.pNext = nullptr;
        viewportState.flags = 0;
        viewportState.viewportCount = 1;
        viewportState.pViewports = nullptr;
        viewportState.scissorCount = 1;
        viewportState.pScissors = nullptr;

        VkPipelineRasterizationStateCreateInfo rasterizationState{};
        rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationState.pNext = nullptr;
        rasterizationState.flags = 0;
        rasterizationState.depthClampEnable = VK_FALSE;
        rasterizationState.rasterizerDiscardEnable = VK_FALSE;
        rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizationState.depthBiasEnable = VK_FALSE;
        rasterizationState.depthBiasConstantFactor = 0.0f;
        rasterizationState.depthBiasClamp = 0.0f;
        rasterizationState.depthBiasSlopeFactor = 0.0f;
        rasterizationState.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampleState{};
        multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleState.pNext = nullptr;
        multisampleState.flags = 0;
        multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampleState.sampleShadingEnable = VK_FALSE;
        multisampleState.minSampleShading = 1.0f;
        multisampleState.pSampleMask = nullptr;
        multisampleState.alphaToCoverageEnable = VK_FALSE;
        multisampleState.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlendAttachmentState{};
        colorBlendAttachmentState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendAttachmentState.pNext = nullptr;
        colorBlendAttachmentState.flags = 0;
        colorBlendAttachmentState.logicOpEnable = VK_FALSE;
        colorBlendAttachmentState.logicOp = VK_LOGIC_OP_COPY;
        colorBlendAttachmentState.attachmentCount = 1;
        colorBlendAttachmentState.pAttachments = &colorBlendAttachment;
        colorBlendAttachmentState.blendConstants[0] = 0.0f;
        colorBlendAttachmentState.blendConstants[1] = 0.0f;
        colorBlendAttachmentState.blendConstants[2] = 0.0f;
        colorBlendAttachmentState.blendConstants[3] = 0.0f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.pNext = nullptr;
        pipelineLayoutInfo.flags = 0;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 0;
        pipelineLayoutInfo.pPushConstantRanges = nullptr;

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("failed to create pipeline layout!");
#ifndef NDEBUG
        std::cerr << "Created pipeline layout " << pipelineLayout << std::endl;
#endif

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext = nullptr;
        pipelineInfo.flags = 0;
        pipelineInfo.stageCount = shaderInfos.size();
        pipelineInfo.pStages = shaderInfos.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pTessellationState = nullptr;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizationState;
        pipelineInfo.pMultisampleState = &multisampleState;
        pipelineInfo.pDepthStencilState = nullptr;
        pipelineInfo.pColorBlendState = &colorBlendAttachmentState;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.basePipelineIndex = - 1;

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline!");
#ifndef NDEBUG
        std::cerr << "Created graphics pipeline " << graphicsPipeline << std::endl;
#endif

        for (VkShaderModule& shaderModule : shaderModules)
            vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    void createFramebuffers()
    {
        swapchainFramebuffers.resize(swapchainImageViews.size());

        std::transform(swapchainImageViews.begin(), swapchainImageViews.end(), swapchainFramebuffers.begin(),
            [this](const VkImageView& imageView)
            {
                VkImageView attachments[] = {
                    imageView
                };

                VkFramebufferCreateInfo framebufferInfo{};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.pNext = nullptr;
                framebufferInfo.flags = 0;
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = attachments;
                framebufferInfo.width = swapchainExtent.width;
                framebufferInfo.height = swapchainExtent.height;
                framebufferInfo.layers = 1;

                VkFramebuffer framebuffer;
                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS)
                    throw std::runtime_error("Failed to create framebuffer!");

                return framebuffer;
            }
            );

#ifndef NDEBUG
        std::cerr << "Created swapchain framebuffers count " << swapchainFramebuffers.size() << std::endl;
#endif
    }

    void createCommandPool()
    {
        QueueFamilyIndices familyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo commandPoolInfo{};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.pNext = nullptr;
        commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolInfo.queueFamilyIndex = familyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create command pool!");
#ifndef NDEBUG
        std::cerr << "Created command pool " << commandPool << std::endl;
#endif
    }

//------------------------------------------------------------
// Buffer creation util functions begin
//------------------------------------------------------------
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        for (uint32_t bit = 0; bit < memoryProperties.memoryTypeCount; ++bit)
            if (typeFilter & (1 << bit) &&
            (memoryProperties.memoryTypes[bit].propertyFlags & properties) == properties)
                return bit;

        throw std::runtime_error("Failed to find suitable memory type!");
    }
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.pNext = nullptr;
        bufferInfo.flags = 0;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.queueFamilyIndexCount = 0;
        bufferInfo.pQueueFamilyIndices = nullptr;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create buffer!");
#ifndef NDEBUG
        std::cerr << "Created buffer " << buffer << std::endl;
#endif

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext = nullptr;
        allocInfo.allocationSize = memoryRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate buffer memory!");
#ifndef NDEBUG
        std::cerr << "Allocated device buffer memory " << bufferMemory << std::endl;
#endif

        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.pNext = nullptr;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pNext = nullptr;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;

        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = nullptr;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = 0;
        submitInfo.pWaitDstStageMask = 0;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = 0;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

        vkDeviceWaitIdle(device);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
//------------------------------------------------------------
// Buffer creation util functions begin
//------------------------------------------------------------

    void createVertexBuffer()
    {
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

#ifndef NDEBUG
        std::cerr << "Created vertex buffer " << vertexBuffer << std::endl;
#endif
    }

    void createIndexBuffer()
    {
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;

        VkDeviceSize bufferSize = sizeof(uint16_t) * indices.size();
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

#ifndef NDEBUG
        std::cerr << "Created index buffer " << indexBuffer << std::endl;
#endif
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(maxFramesInFlight);
        uniformBuffersMemory.resize(maxFramesInFlight);
        uniformBuffersMapped.resize(maxFramesInFlight);

        for (int i = 0; i < maxFramesInFlight; ++i)
        {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }

#ifndef NDEBUG
        std::cerr << "Created uniform buffers count " << uniformBuffers.size() << std::endl;
#endif
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(maxFramesInFlight);

        VkDescriptorPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        createInfo.pNext = nullptr;
        createInfo.flags = 0;
        createInfo.maxSets = static_cast<uint32_t>(maxFramesInFlight);
        createInfo.poolSizeCount = 1;
        createInfo.pPoolSizes = &poolSize;

        if (vkCreateDescriptorPool(device, &createInfo, nullptr, &descriptorPool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor pool!");
#ifndef NDEBUG
        std::cerr << "Created descriptor pool " << descriptorPool << std::endl;
#endif
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.pNext = nullptr;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(maxFramesInFlight);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(maxFramesInFlight);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to create descriptor sets!");
#ifndef NDEBUG
        std::cerr << "Created descriptor sets count  " << descriptorSets.size() << std::endl;
#endif

        for (int i = 0; i < maxFramesInFlight; ++i)
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = VK_WHOLE_SIZE;

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.pNext = nullptr;
            descriptorWrite.dstSet = descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.pImageInfo = nullptr;
            descriptorWrite.pBufferInfo = &bufferInfo;
            descriptorWrite.pTexelBufferView = nullptr;

            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(maxFramesInFlight);

        VkCommandBufferAllocateInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.pNext = nullptr;
        commandBufferInfo.commandPool = commandPool;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = maxFramesInFlight;

        if (vkAllocateCommandBuffers(device, &commandBufferInfo, commandBuffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate command buffer!");
#ifndef NDEBUG
        std::cerr << "Allocated command buffer count " << maxFramesInFlight << std::endl;
#endif
    }

    void createSyncObjects()
    {
        imageAvaliableSemaphores.resize(maxFramesInFlight);
        renderFinishedSemaphores.resize(maxFramesInFlight);
        inFlightFences.resize(maxFramesInFlight);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreInfo.pNext = nullptr;
        semaphoreInfo.flags = 0;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.pNext = nullptr;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (VkSemaphore& imageAvaliableSemaphore : imageAvaliableSemaphores)
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvaliableSemaphore) != VK_SUCCESS)
                throw("Failed to create image available semaphore!");
#ifndef NDEBUG
        std::cerr << "Created image available semaphores count " << imageAvaliableSemaphores.size() << std::endl;
#endif

        for (VkSemaphore& renderFinishedSemaphore : renderFinishedSemaphores)
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS)
                throw("Failed to create render finished semaphore!");
#ifndef NDEBUG
        std::cerr << "Created render finished semaphores count " << renderFinishedSemaphores.size() << std::endl;
#endif

        for (VkFence& inFlightFence : inFlightFences)
            if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS)
                throw("Failed to create in flight fence!");
#ifndef NDEBUG
        std::cerr << "Created in flight fences count " << inFlightFences.size() << std::endl;
#endif
    }

//------------------------------------------------------------
// Main loop functions
//------------------------------------------------------------

//------------------------------------------------------------
// Draw frame util functions begin
//------------------------------------------------------------
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pNext = nullptr;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin recording command buffer!");

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.pNext = nullptr;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapchainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchainExtent;
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 1.0f;
        viewport.width = (float)swapchainExtent.width;
        viewport.height = (float)swapchainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
            throw std::runtime_error("failed to record command buffer!");
    }
    void updateUniformBuffer(uint32_t currentImage)
    {
        static std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

        std::chrono::high_resolution_clock::time_point currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        proj = glm::perspective(glm::radians(60.0f), (float)swapchainExtent.width / swapchainExtent.height, 0.1f, 10.f);
        proj[1][1] *= - 1;

        ubo.mvp = proj * model * view;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }
//------------------------------------------------------------
// Draw frame util functions end
//------------------------------------------------------------

    void drawFrame()
    {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t imageIndex = 0;
        VkResult result = vkAcquireNextImageKHR(device, swapchain, std::numeric_limits<uint64_t>::max(), imageAvaliableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
            recreateSwapchain();
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Failed to acquire swap chain image!");

        updateUniformBuffer(currentFrame);
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        VkSemaphore waitSemaphores[] = {imageAvaliableSemaphores[currentFrame]};
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = nullptr;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("Failed to submit graphics queue!");

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = nullptr;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            framebufferResized = false;
            recreateSwapchain();
        }
        else if (result != VK_SUCCESS)
            throw std::runtime_error("Failed to present swap chain image!");

        currentFrame = (currentFrame + 1) % maxFramesInFlight;
    }
};



int main(int argc, char* argv[])
{
    HelloTriangleApplication app;

    try
    {
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
