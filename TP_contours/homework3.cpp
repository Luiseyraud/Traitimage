#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// lot d'images et contours de reference
std::vector<std::string> noms;
std::vector<cv::Mat> images;
std::vector<cv::Mat> references;

// detecteur choisi par les boutons
int detecteur = 0;   // 0 = Sobel, 1 = Laplace, 2 = Canny

// parametres reglables par les curseurs
int numImage = 0;
int flou = 2;        // sigma du flou gaussien = flou / 2 (0 = pas de flou)
int noyau = 1;       // Sobel et Laplace : taille du noyau = 2 * noyau + 1
int seuil = 50;      // Sobel et Laplace : seuil sur la norme ramenee a [0,255], Canny : seuil bas
int seuilHaut = 150; // Canny : seuil haut

const std::string nomsDetecteurs[3] = {"Sobel", "Laplace", "Canny"};

// une seule fenetre : bandeau de boutons, puis image, reference et contours cote a cote
const std::string nomFenetre = "Detection de contours";
const int hauteurBandeau = 50;
const cv::Rect boutons[3] = {cv::Rect(10, 8, 150, 34), cv::Rect(170, 8, 150, 34), cv::Rect(330, 8, 150, 34)};

// curseurs dessines sous les images (une trackbar OpenCV ne peut pas etre masquee)
struct Curseur {
    std::string nom;
    int* valeur;
    int max;
};
const int hauteurCurseur = 32;
const int debutCurseur = 250;   // abscisse du debut de la barre
int curseurActif = -1;          // curseur en cours de deplacement

struct Mesures {
    int detectes, reference, corrects, fauxPositifs, fauxNegatifs;
    double P, TFP, TFN;
};

cv::Mat detecterContours(const cv::Mat& image)
{
    cv::Mat lisse;
    if (flou > 0)
        cv::GaussianBlur(image, lisse, cv::Size(), flou / 2.0);
    else
        lisse = image;

    int taille = 2 * noyau + 1;
    cv::Mat contours;

    if (detecteur == 0) {
        // norme du gradient de Sobel
        cv::Mat sobelX, sobelY, norme, orientation;
        cv::Sobel(lisse, sobelX, CV_32F, 1, 0, taille);
        cv::Sobel(lisse, sobelY, CV_32F, 0, 1, taille);
        cv::cartToPolar(sobelX, sobelY, norme, orientation);

        double minval, maxval;
        cv::minMaxLoc(norme, &minval, &maxval);
        cv::Mat sobelImage;
        norme.convertTo(sobelImage, CV_8U, 255 / maxval);
        cv::threshold(sobelImage, contours, seuil, 255, cv::THRESH_BINARY);
    }
    else if (detecteur == 1) {
        // passages par zero du laplacien
        cv::Mat laplace;
        cv::Laplacian(lisse, laplace, CV_32F, taille);

        cv::Mat signImage, binary, dilated;
        cv::threshold(laplace, signImage, 0, 255, cv::THRESH_BINARY);
        signImage.convertTo(binary, CV_8U);
        cv::dilate(binary, dilated, cv::Mat());
        cv::Mat zeros = dilated - binary;

        // on ne garde que les passages par zero de forte pente
        cv::Mat pente;
        cv::morphologyEx(laplace, pente, cv::MORPH_GRADIENT, cv::Mat());

        double minval, maxval;
        cv::minMaxLoc(pente, &minval, &maxval);
        cv::Mat penteImage;
        pente.convertTo(penteImage, CV_8U, 255 / maxval);
        cv::threshold(penteImage, contours, seuil, 255, cv::THRESH_BINARY);
        cv::bitwise_and(contours, zeros, contours);
    }
    else {
        cv::Canny(lisse, contours, seuil, seuilHaut);
    }
    return contours;
}

Mesures comparer(const cv::Mat& contours, const cv::Mat& reference)
{
    // voisinage 3x3 : dilatation par l'element structurant par defaut
    cv::Mat referenceDilatee, contoursDilates;
    cv::dilate(reference, referenceDilatee, cv::Mat());
    cv::dilate(contours, contoursDilates, cv::Mat());

    // pixels detectes ayant un pixel de reference dans leur voisinage
    cv::Mat corrects;
    cv::bitwise_and(contours, referenceDilatee, corrects);

    // pixels de reference sans pixel detecte dans leur voisinage
    cv::Mat manques;
    cv::bitwise_and(reference, 255 - contoursDilates, manques);

    Mesures m;
    m.detectes = cv::countNonZero(contours);
    m.reference = cv::countNonZero(reference);
    m.corrects = cv::countNonZero(corrects);
    m.fauxPositifs = m.detectes - m.corrects;
    m.fauxNegatifs = cv::countNonZero(manques);

    double total = m.corrects + m.fauxPositifs + m.fauxNegatifs;
    m.P = m.corrects / total;
    m.TFP = m.fauxPositifs / total;
    m.TFN = m.fauxNegatifs / total;
    return m;
}

void afficherEntete()
{
    std::cout << std::left << std::setw(14) << "image" << std::right
              << std::setw(10) << "detectes"
              << std::setw(10) << "reference"
              << std::setw(10) << "corrects"
              << std::setw(10) << "faux_pos"
              << std::setw(10) << "faux_neg"
              << std::setw(8) << "P"
              << std::setw(8) << "TFP"
              << std::setw(8) << "TFN" << std::endl;
}

void afficherMesures(const std::string& nom, const Mesures& m)
{
    std::cout << std::left << std::setw(14) << nom << std::right
              << std::setw(10) << m.detectes
              << std::setw(10) << m.reference
              << std::setw(10) << m.corrects
              << std::setw(10) << m.fauxPositifs
              << std::setw(10) << m.fauxNegatifs
              << std::fixed << std::setprecision(3)
              << std::setw(8) << m.P
              << std::setw(8) << m.TFP
              << std::setw(8) << m.TFN << std::endl;
}

void traiterLot()
{
    std::cout << std::endl << nomsDetecteurs[detecteur]
              << std::fixed << std::setprecision(1) << " (flou " << flou / 2.0 << ", noyau " << 2 * noyau + 1
              << ", seuil " << seuil << ", seuil haut " << seuilHaut << ")" << std::endl;
    afficherEntete();

    double sommeP = 0, sommeTFP = 0, sommeTFN = 0;
    for (size_t i = 0; i < images.size(); i++) {
        Mesures m = comparer(detecterContours(images[i]), references[i]);
        afficherMesures(noms[i], m);
        sommeP += m.P;
        sommeTFP += m.TFP;
        sommeTFN += m.TFN;
    }

    int n = images.size();
    std::cout << std::left << std::setw(64) << "moyenne" << std::right
              << std::setprecision(3)
              << std::setw(8) << sommeP / n
              << std::setw(8) << sommeTFP / n
              << std::setw(8) << sommeTFN / n << std::endl;
}

// seuls les parametres utilises par le detecteur choisi ont un curseur
std::vector<Curseur> curseurs()
{
    std::vector<Curseur> c = {{"Image", &numImage, static_cast<int>(images.size()) - 1}, {"Flou", &flou, 10}};
    if (detecteur != 2)   // Sobel et Laplace
        c.push_back({"Noyau", &noyau, 3});
    c.push_back({"Seuil", &seuil, 255});
    if (detecteur == 2)   // Canny
        c.push_back({"Seuil haut", &seuilHaut, 500});
    return c;
}

// ordonnee du centre du curseur i et abscisse de la fin des barres
int ligneCurseur(int i)
{
    return hauteurBandeau + images[numImage].rows + hauteurCurseur * i + hauteurCurseur / 2 + 5;
}

int finCurseur()
{
    return 3 * images[numImage].cols - 25;
}

void afficher()
{
    const cv::Mat& image = images[numImage];
    cv::Mat contours = detecterContours(image);
    std::vector<Curseur> c = curseurs();

    cv::Mat fenetre(hauteurBandeau + image.rows + hauteurCurseur * c.size() + 10, 3 * image.cols, CV_8U, cv::Scalar(255));
    cv::Mat imageROI;
    imageROI = fenetre(cv::Rect(0, hauteurBandeau, image.cols, image.rows));
    image.copyTo(imageROI);
    imageROI = fenetre(cv::Rect(image.cols, hauteurBandeau, image.cols, image.rows));
    cv::Mat(255 - references[numImage]).copyTo(imageROI);
    imageROI = fenetre(cv::Rect(2 * image.cols, hauteurBandeau, image.cols, image.rows));
    cv::Mat(255 - contours).copyTo(imageROI);

    cv::Mat affichage;
    cv::cvtColor(fenetre, affichage, cv::COLOR_GRAY2BGR);

    // boutons, celui du detecteur choisi en vert
    for (int i = 0; i < 3; i++) {
        bool actif = (i == detecteur);
        cv::rectangle(affichage, boutons[i],
                      actif ? cv::Scalar(0, 160, 0) : cv::Scalar(200, 200, 200), cv::FILLED);
        cv::putText(affichage, nomsDetecteurs[i], boutons[i].tl() + cv::Point(15, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    actif ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0), 2);
    }
    cv::putText(affichage, "Reference", cv::Point(image.cols + 10, 33),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(affichage, "Contours", cv::Point(2 * image.cols + 10, 33),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

    // curseurs : nom, valeur, barre et poignee
    for (size_t i = 0; i < c.size(); i++) {
        int y = ligneCurseur(i);
        int x = debutCurseur + *c[i].valeur * (finCurseur() - debutCurseur) / std::max(1, c[i].max);
        std::string valeur = (c[i].valeur == &numImage) ? noms[numImage] : std::to_string(*c[i].valeur);
        cv::putText(affichage, c[i].nom, cv::Point(10, y + 6), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::putText(affichage, valeur, cv::Point(120, y + 6), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        cv::line(affichage, cv::Point(debutCurseur, y), cv::Point(finCurseur(), y), cv::Scalar(200, 200, 200), 4);
        cv::line(affichage, cv::Point(debutCurseur, y), cv::Point(x, y), cv::Scalar(0, 160, 0), 4);
        cv::circle(affichage, cv::Point(x, y), 9, cv::Scalar(0, 160, 0), cv::FILLED);
    }

    cv::imshow(nomFenetre, affichage);

    std::cout << nomsDetecteurs[detecteur] << " : ";
    afficherMesures(noms[numImage], comparer(contours, references[numImage]));
}

void onMouse(int event, int x, int y, int, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN) {
        for (int i = 0; i < 3; i++)
            if (boutons[i].contains(cv::Point(x, y)) && i != detecteur) {
                detecteur = i;
                afficher();
            }

        // clic sur un curseur : il suit la souris jusqu'au relachement du bouton
        curseurActif = -1;
        for (size_t i = 0; i < curseurs().size(); i++)
            if (std::abs(y - ligneCurseur(i)) <= hauteurCurseur / 2 && x >= debutCurseur - 10 && x <= finCurseur() + 10)
                curseurActif = i;
    }
    else if (event == cv::EVENT_LBUTTONUP)
        curseurActif = -1;

    if (curseurActif < 0 || (event != cv::EVENT_LBUTTONDOWN && event != cv::EVENT_MOUSEMOVE))
        return;

    Curseur c = curseurs()[curseurActif];
    int valeur = cvRound(static_cast<double>(x - debutCurseur) * c.max / (finCurseur() - debutCurseur));
    valeur = std::min(std::max(valeur, 0), c.max);
    if (valeur != *c.valeur) {
        *c.valeur = valeur;
        afficher();
    }
}

bool fenetreOuverte()
{
    // avec Qt, interroger une fenetre fermee leve une exception
    try {
        return cv::getWindowProperty(nomFenetre, cv::WND_PROP_VISIBLE) > 0;
    }
    catch (const cv::Exception&) {
        return false;
    }
}

void chargerLot(const std::string& dossier)
{
    std::vector<cv::String> fichiers;
    cv::glob(dossier + "/*.pgm", fichiers);

    for (const cv::String& fichier : fichiers) {
        std::string nom = fichier.substr(fichier.find_last_of('/') + 1);
        nom = nom.substr(0, nom.size() - 4);

        cv::Mat image = cv::imread(fichier, cv::IMREAD_GRAYSCALE);
        cv::Mat gt = cv::imread(dossier + "/gt/" + nom + "_gt_binary.pgm", cv::IMREAD_GRAYSCALE);
        if (image.empty() || gt.empty())
            continue;

        // contours de reference noirs sur fond blanc -> contours a 255
        cv::Mat reference;
        cv::threshold(gt, reference, 128, 255, cv::THRESH_BINARY_INV);

        noms.push_back(nom);
        images.push_back(image);
        references.push_back(reference);
    }
}

int main(int argc, char** argv)
{
    std::string dossier = argc > 1 ? argv[1] : "images";
    chargerLot(dossier);

    if (images.empty()) {
        std::cout << "Aucune image trouvee dans " << dossier << std::endl;
        return 0;
    }

    cv::namedWindow(nomFenetre);
    cv::setMouseCallback(nomFenetre, onMouse);

    std::cout << "l : traiter le lot, q : quitter" << std::endl;
    afficher();

    // attente par pas de 50 ms pour quitter aussi quand la fenetre est fermee
    while (fenetreOuverte()) {
        int touche = cv::waitKey(50);
        if (touche == 'l')
            traiterLot();
        else if (touche == 'q')
            break;
    }
    return 0;
}
