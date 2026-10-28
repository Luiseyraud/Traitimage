// Balayage automatique des parametres sur le lot d'images.
// Reutilise les fonctions de homework3.cpp (detecterContours, comparer, chargerLot).
// Compilation : g++ -O2 -fopenmp balayage.cpp -o balayage $(pkg-config --cflags --libs opencv4)
// Utilisation : ./balayage ../images > balayage.csv
#define main main_homework3
#include "../homework3.cpp"
#undef main

void evaluer()
{
    int n = images.size();
    std::vector<Mesures> resultats(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        resultats[i] = comparer(detecterContours(images[i]), references[i]);

    for (int i = 0; i < n; i++) {
        const Mesures& m = resultats[i];
        std::cout << nomsDetecteurs[detecteur] << "," << flou << "," << 2 * noyau + 1 << ","
                  << seuil << "," << seuilHaut << "," << noms[i] << ","
                  << m.detectes << "," << m.reference << "," << m.corrects << ","
                  << m.fauxPositifs << "," << m.fauxNegatifs << ","
                  << m.P << "," << m.TFP << "," << m.TFN << "\n";
    }
}

int main(int argc, char** argv)
{
    chargerLot(argc > 1 ? argv[1] : "../images");
    cv::setNumThreads(1);
    std::cout << std::setprecision(5)
              << "detecteur,flou,noyau,seuil,seuil_haut,image,detectes,reference,corrects,"
              << "faux_positifs,faux_negatifs,P,TFP,TFN\n";

    // Sobel et Laplace : flou, taille du noyau et seuil
    for (detecteur = 0; detecteur < 2; detecteur++)
        for (flou = 0; flou <= 8; flou++)
            for (noyau = 0; noyau <= 3; noyau++)
                for (seuil = 5; seuil <= 200; seuil += 5)
                    evaluer();

    // Canny : flou, seuil bas et seuil haut
    detecteur = 2;
    noyau = 1;
    for (flou = 0; flou <= 8; flou++)
        for (seuil = 10; seuil <= 300; seuil += 10)
            for (seuilHaut = seuil; seuilHaut <= 500; seuilHaut += 20)
                evaluer();
    return 0;
}
