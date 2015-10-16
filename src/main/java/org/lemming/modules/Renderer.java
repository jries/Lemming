package org.lemming.modules;

import java.util.List;

import ij.ImagePlus;

import org.lemming.pipeline.MultiRunModule;
import org.lemming.interfaces.Element;

public abstract class Renderer extends MultiRunModule {
	
	
	protected ImagePlus ip;
	final protected String title = "LemMING!"; // title of the image

	public Renderer() {
		ip = new ImagePlus();
		ip.setTitle(title);
	}
	
	public ImagePlus getImage(){
		return ip;
	}
	
	public void resetInputStore(){
		inputs.clear();
		iterator=null;
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
	}
	
	public abstract void preview(List<Element> previewList);

}
