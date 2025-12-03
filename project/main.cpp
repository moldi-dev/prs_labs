#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "src/utils/BarcodeUtils.h"

namespace fs = std::filesystem;

// Logging system (redirect from the console to the application console logs)
struct AppLog {
    ImGuiTextBuffer Buf;
    ImVector<int> LineOffsets;
    bool AutoScroll;
    bool ScrollToBottom;

    AppLog() {
        AutoScroll = true;
        ScrollToBottom = false;
        Clear();
    }

    void Clear() {
        Buf.clear();
        LineOffsets.clear();
        LineOffsets.push_back(0);
    }

    void AddLog(const char *fmt, ...) IM_FMTARGS(2) {
        int old_size = Buf.size();
        va_list args;
        va_start(args, fmt);
        Buf.appendfv(fmt, args);
        va_end(args);
        for (int new_size = Buf.size(); old_size < new_size; old_size++) {
            if (Buf[old_size] == '\n') {
                LineOffsets.push_back(old_size + 1);
            }
        }

        if (AutoScroll) {
            ScrollToBottom = true;
        }
    }

    void Draw(const char *title, bool *p_open = NULL) {
        if (!ImGui::Begin(title, p_open)) {
            ImGui::End();
            return;
        }

        if (ImGui::Button("Clear")) {
            Clear();
        }

        ImGui::SameLine();
        ImGui::Checkbox("Auto-scroll", &AutoScroll);
        ImGui::Separator();

        // Reserve space for the scroll region
        ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::TextUnformatted(Buf.begin());

        if (ScrollToBottom) {
            ImGui::SetScrollHereY(1.0f);
            ScrollToBottom = false;
        }

        ImGui::EndChild();
        ImGui::End();
    }
};

// Global instance so streambuf can find it
static AppLog g_AppLog;

// Custom stream buffer to redirect std::cout to AppLog
class ConsoleRedirector : public std::streambuf {
public:
    int overflow(int c) override {
        if (c != EOF) {
            char ch = (char) c;
            g_AppLog.AddLog("%c", ch);
        }

        return c;
    }

    std::streamsize xsputn(const char *s, std::streamsize n) override {
        std::string str(s, n);
        g_AppLog.AddLog("%s", str.c_str());
        return n;
    }
};

// GPU Texture
struct GLTexture {
    GLuint id = 0;
    int width = 0;
    int height = 0;

    void Update(const cv::Mat &mat) {
        if (mat.empty()) {
            return;
        }

        if (id == 0) {
            glGenTextures(1, &id);
            glBindTexture(GL_TEXTURE_2D, id);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        glBindTexture(GL_TEXTURE_2D, id);
        glPixelStorei(GL_UNPACK_ALIGNMENT, (mat.step & 3) ? 1 : 4);

        GLenum format = GL_BGR;

        if (mat.channels() == 4) {
            format = GL_BGRA;
        }

        else if (mat.channels() == 1) {
            format = GL_LUMINANCE;
        }

        if (width != mat.cols || height != mat.rows) {
            width = mat.cols;
            height = mat.rows;
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, format, GL_UNSIGNED_BYTE, mat.data);
        }

        else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, GL_UNSIGNED_BYTE, mat.data);
        }
    }
};


int main(int, char **) {
    // Setup Redirector
    ConsoleRedirector redirector;
    std::streambuf *oldCoutStream = std::cout.rdbuf(&redirector);
    std::streambuf *oldCerrStream = std::cerr.rdbuf(&redirector);

    // Setup Window
    glfwSetErrorCallback([](int error, const char *description) {
        fprintf(stderr, "Glfw Error %d: %s\n", error, description);
    });

    if (!glfwInit()) {
        return 1;
    }

    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Initial window size (Desktop window, not ImGui window)
    GLFWwindow *window = glfwCreateWindow(1600, 900, "Barcode Scanner", NULL, NULL);

    if (window == NULL) {
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 4.0f;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    std::cout << "System Ready." << std::endl;
    std::cout << "Input Directory: ./assets/sample_images/" << std::endl;

    char inputBuf[128] = "image_1.jpg";
    bool verbose = true;
    int currentTab = 0;

    cv::Mat imgOriginal, imgPre, imgEdges, imgBox, imgExtractedBarcode, imgDetectedBarcode;
    GLTexture displayTexture;
    bool hasRun = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        float width = (float) display_w;
        float height = (float) display_h;

        // Controls panel
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 400), ImGuiCond_FirstUseEver);

        ImGui::Begin("Controls");
        ImGui::TextDisabled("Input Settings");
        ImGui::InputText("Filename", inputBuf, IM_ARRAYSIZE(inputBuf));

        if (ImGui::Button("Load Image", ImVec2(-1, 0))) {
            std::string path = std::string("./assets/sample_images/") + inputBuf;
            std::cout << "[IO] Loading: " << path << std::endl;

            if (fs::exists(path)) {
                imgOriginal = cv::imread(path);

                if (!imgOriginal.empty()) {
                    std::cout << "[IO] Loaded " << imgOriginal.cols << "x" << imgOriginal.rows << std::endl;
                    displayTexture.Update(imgOriginal);
                    hasRun = false;
                    currentTab = 0;
                }

                else {
                    std::cerr << "[Error] Failed to decode image." << std::endl;
                }
            }

            else {
                std::cerr << "[Error] File not found." << std::endl;
            }
        }

        ImGui::Checkbox("Verbose Mode", &verbose);
        ImGui::Separator();

        // Run button
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4) ImColor::HSV(0.6f, 0.6f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4) ImColor::HSV(0.6f, 0.7f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4) ImColor::HSV(0.6f, 0.8f, 0.8f));

        if (ImGui::Button("RUN PIPELINE", ImVec2(-1, 20))) {
            if (imgOriginal.empty()) {
                std::cerr << "[Error] No image loaded." << std::endl;
            }

            else {
                std::cout << "Starting the pipeline..." << std::endl;

                BarcodeDetector detector(verbose);
                imgDetectedBarcode = detector.scan(imgOriginal);
                hasRun = true;

                if (verbose) {
                    std::cout << "[GUI] Fetching debug layers..." << std::endl;
                    imgPre = cv::imread("./assets/results/1_preprocessed.jpg");
                    imgEdges = cv::imread("./assets/results/2_edges.jpg");
                    imgBox = cv::imread("./assets/results/3_bounding_box.jpg");
                    imgExtractedBarcode = cv::imread("./assets/results/4_final_crop.jpg");
                    imgDetectedBarcode = cv::imread("./assets/results/5_decoded_result.jpg");
                }

                currentTab = verbose ? 4 : 0;
            }
        }

        ImGui::PopStyleColor(3);

        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4) ImColor::HSV(0.6f, 0.6f, 0.6f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4) ImColor::HSV(0.6f, 0.7f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4) ImColor::HSV(0.6f, 0.8f, 0.8f));

        // OCR Button
        if (ImGui::Button("RUN OCR", ImVec2(-1, 20))) {
            std::cout << "\n[OCR] Initializing Tesseract..." << std::endl;
            std::cout << "[OCR] Error: OCR module not fully implemented in this build." << std::endl;
        }

        ImGui::PopStyleColor(3);

        ImGui::End();

        // Viewport
        ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(width - 330, height * 0.8f), ImGuiCond_FirstUseEver);

        ImGui::Begin("Viewport");

        // Tabs
        if (hasRun && verbose) {
            if (ImGui::BeginTabBar("ImageTabs")) {
                if (ImGui::BeginTabItem("Original")) {
                    currentTab = 0;
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Preprocessed")) {
                    currentTab = 1;
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Edges")) {
                    currentTab = 2;
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Region")) {
                    currentTab = 3;
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Extracted Barcode")) {
                    currentTab = 4;
                    ImGui::EndTabItem();
                }

                if (ImGui::BeginTabItem("Detected EAN-13 Barcode")) {
                    currentTab = 5;
                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }

        // Image selection
        cv::Mat *matToShow = nullptr;
        if (!imgOriginal.empty()) {
            if (!hasRun) {
                matToShow = &imgOriginal;
            }

            else if (!verbose) {
                matToShow = imgDetectedBarcode.empty() ? &imgOriginal : &imgDetectedBarcode;
            }

            else {
                switch (currentTab) {
                    case 0:
                        matToShow = &imgOriginal;
                        break;
                    case 1:
                        matToShow = &imgPre;
                        break;
                    case 2:
                        matToShow = &imgEdges;
                        break;
                    case 3:
                        matToShow = &imgBox;
                        break;
                    case 4:
                        matToShow = &imgExtractedBarcode;
                        break;
                    case 5:
                        matToShow = &imgDetectedBarcode;
                        break;
                    default:
                        break;
                }
            }
        }

        // Render Logic
        if (matToShow && !matToShow->empty()) {
            displayTexture.Update(*matToShow);

            ImVec2 avail = ImGui::GetContentRegionAvail();
            avail.y -= 25;

            // Aspect ratio
            float imgAspect = (float) matToShow->cols / (float) matToShow->rows;
            float winAspect = avail.x / avail.y;

            float drawW, drawH;

            if (imgAspect > winAspect) {
                // Fit width
                drawW = avail.x;
                drawH = drawW / imgAspect;
            }

            else {
                // Fit height
                drawH = avail.y;
                drawW = drawH * imgAspect;
            }

            float cursorX = (avail.x - drawW) * 0.5f;
            float cursorY = (avail.y - drawH) * 0.5f;
            ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + cursorX, ImGui::GetCursorPosY() + cursorY));

            ImGui::Image(displayTexture.id, ImVec2(drawW, drawH));

            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
            ImGui::Separator();
            ImGui::Text("Resolution: %dx%d  |  Channels: %d", matToShow->cols, matToShow->rows, matToShow->channels());
        }

        else {
            ImGui::SetCursorPos(ImVec2(ImGui::GetContentRegionAvail().x * 0.45f,
                                       ImGui::GetContentRegionAvail().y * 0.5f));
            ImGui::TextDisabled("No Image Loaded");
        }

        ImGui::End();

        // Console log (bottom panel)
        ImGui::SetNextWindowPos(ImVec2(10, height * 0.68f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(width - 20, height * 0.30f), ImGuiCond_FirstUseEver);

        g_AppLog.Draw("Console Log");

        // Rendering
        ImGui::Render();
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup & Restore cout
    std::cout.rdbuf(oldCoutStream);
    std::cerr.rdbuf(oldCerrStream);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}