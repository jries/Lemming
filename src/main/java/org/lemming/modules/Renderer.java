package org.lemming.modules;

import ij.ImagePlus;

import org.lemming.pipeline.MultiRunModule;

public abstract class Renderer extends MultiRunModule {
	
	
	protected ImagePlus ip;
	final protected String title = "LemMING!"; // title of the image

	public Renderer() {
		ip = new ImagePlus();
	}
	
	public ImagePlus getImage(){
		return ip;
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
	}

}
