package com.yucl.demo.djl.liveobjectdetect;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.WindowConstants;

public class ViewerFrame {

    private JFrame frame;
    private ImagePanel imagePanel;

    ViewerFrame(int width, int height) {
        frame = new JFrame("Demo");
        imagePanel = new ImagePanel();
        frame.setLayout(new BorderLayout());
        frame.add(BorderLayout.CENTER, imagePanel);

        JOptionPane.setRootFrame(frame);
        Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
        if (width > screenSize.width) {
            width = screenSize.width;
        }
        Dimension frameSize = new Dimension(width, height);
        frame.setSize(frameSize);
        frame.setLocation((screenSize.width - width) / 2, (screenSize.height - height) / 2);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    void showImage(BufferedImage image) {
        imagePanel.setImage(image);
        SwingUtilities.invokeLater(
                () -> {
                    frame.repaint();
                    frame.pack();
                });
    }

    private static final class ImagePanel extends JPanel {

        private BufferedImage image;

        void setImage(BufferedImage image) {
            this.image = image;
        }

        @Override
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (image == null) {
                return;
            }

            g.drawImage(image, 0, 0, null);
            setPreferredSize(new Dimension(image.getWidth(), image.getHeight()));
        }
    }
}