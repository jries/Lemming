package org.lemming.gui;

import javax.swing.SwingUtilities;
import ij.IJ;
import ij.ImagePlus;
import ij.plugin.PlugIn;

public class Lemming implements PlugIn {

	@Override
	public void run(String arg) {
		final ImagePlus imp = IJ.getImage();
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				try {
					Controller frame = new Controller(imp);
					frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

	public static void main(String[] args) {
		final ImagePlus imp = IJ.createImage("Test", 320, 320, 1, 8);
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				try {
					Controller frame = new Controller(imp);
					frame.setVisible(true);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		});
	}

}
