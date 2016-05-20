package org.lemming.modules;

import java.util.List;

import ij.ImagePlus;

import org.lemming.pipeline.MultiRunModule;
import org.lemming.interfaces.Element;

/**
 * base class for all renderer plug-ins
 * 
 * @author Ronny Sczech
 *
 */
public abstract class Renderer extends MultiRunModule {
	
	protected final ImagePlus ip;

	protected Renderer() {
		ip = new ImagePlus();
		String title = "Renderer Window";
		ip.setTitle(title);
	}
	
	public ImagePlus getImage(){
		return ip;
	}
	
	@Override
	public boolean check() {
		return inputs.size()==1;
	}
	
	@Override
	public void afterRun(){
		double max = ip.getStatistics().histMax;
		ip.getProcessor().setMinAndMax(0, max);
		ip.updateAndRepaintWindow();
		System.out.println("Rendering done in "
				+ (System.currentTimeMillis() - start) + "ms.");
	}
	
	public void show(){
		if (ip!=null){
			ip.show();
			while (ip.isVisible())
				pause(10);
		}
	}
	
	public void preview(List<Element> previewList) {
		for(Element l:previewList) newOutput(l);
		double max = ip.getStatistics().histMax;
		ip.getProcessor().setMinAndMax(0, max);
		ip.updateAndRepaintWindow();
	}

}
