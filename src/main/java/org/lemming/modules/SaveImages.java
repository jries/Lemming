package org.lemming.modules;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;

import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.NumericType;

import org.lemming.pipeline.Element;
import org.lemming.pipeline.Frame;
import org.lemming.pipeline.SingleRunModule;

public class SaveImages<T extends NumericType<T>, F extends Frame<T>> extends SingleRunModule {
	
	private String filename;
	private ImageStack stack;

	public SaveImages(String filename){
		this.filename = filename;
	}
	
	@SuppressWarnings("unchecked")
	@Override
	protected void beforeRun(){
		iterator = inputs.keySet().iterator().next();
		F frame = (F) inputs.get(iterator).peek();
		stack = ImageJFunctions.wrap(frame.getPixels(), "").createEmptyStack();
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public void process(Element data) {
		F frame = (F) data;
		if (frame == null) return;
		stack.addSlice(ImageJFunctions.wrap(frame.getPixels(), "" + frame.getFrameNumber()).getProcessor());
		if (frame.isLast()){ // make the poison pill
			System.out.println("Last Frame saved:" +  frame.getFrameNumber());
			cancel();
		}
	}
	
	@Override
	protected void afterRun(){
		ImagePlus out = new ImagePlus();
		out.setOpenAsHyperStack(true);
		out.setStack(stack, 1, 1, stack.getSize());
		IJ.save(out, filename);
	}

}
