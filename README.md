# Vision Sentinel — Reconnaissance faciale fiable + réglage du seuil
- **Seuil de similarité** (cosine) **réglable** dans l'UI (0.20–0.80). Par défaut **0.35**.
- **Affichage de la similarité** à côté du nom (toggle dans `config.yaml`).
- Boutons **Recharger galerie** et **Capturer x10** pour des gabarits plus robustes.
- Toujours : **zones multiples**, **lignes virtuelles A→B/B→A**, **objets/vêtements**.

- Ajout d'un fichier `sources.json` pour ajouter des flux vidéo dans la liste

## Installation
pip install -r requirements.txt

## Lancement (recommandé)
python -m app.main

## Conseils pour de bons matchs
- Éclairez bien le visage, évitez les contre-jours; regardez la caméra.
- Faites **plusieurs captures** (bouton *Capturer x10*) sous des angles/expressions variés.
- Ajustez le **seuil** :
  - 0.30–0.36 : tolérant (moins de “Unknown”).  
  - 0.38–0.45 : plus strict (moins de faux positifs).
- Vous pouvez passer à `yolo.model: yolov8s.pt` pour une détection plus précise.

## Détection arme à feu
- Utilisez le modèle `best.pt`

## RGPD
Informez, limitez, sécurisez, documentez. Le floutage des inconnus est disponible dans l’UI.
