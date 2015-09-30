package org.lemming.modules;

import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicInteger;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Frame;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.CircularFifoQueue;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.FittedLocalization;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.MultiRunModule;

import ij.gui.Roi;

public abstract class Fitter2<T extends RealType<T>, F extends Frame<T>> extends MultiRunModule {

	private long start;
	protected int size;
	private Store locInput;
	private CircularFifoQueue<Element> firstQueue;
	private Store secondQueue;
	private int queueSize;
	private AtomicInteger counter = new AtomicInteger(0);

	public Fitter2(int queueSize, int windowSize) {
		this.size = windowSize;
		this.queueSize = queueSize;
		firstQueue = queueSize == 0 ? new CircularFifoQueue<Element>(128) : new CircularFifoQueue<Element>(queueSize);
		secondQueue = new FastStore();
	}

	public long getWindowSize(){
		return size;
	}

	@Override
	protected void beforeRun() {
		Iterator<Integer> it = inputs.keySet().iterator();
		try {
			iterator = it.next(); 							// first input
			locInput = inputs.get(it.next()); 				// second input
		} catch (NoSuchElementException | NullPointerException ex){
			System.err.println("Input provided not correct!");
			Thread.currentThread().interrupt();
		}
		
		start = System.currentTimeMillis();
		while (locInput.getLength()<queueSize) pause(10);
		fillQueue();
	}
	
	private void fillQueue(){
		for (int i=firstQueue.size();i<queueSize;i++){
			Element el = locInput.get();
			if (el instanceof FrameElements) 
				firstQueue.add(el);		
		}
	}

	@SuppressWarnings("unchecked")
	@Override
	public Element process(Element data) {
		F frame = (F) data;
		if (frame == null)
			return null;

		if (frame.isLast()) {
			if (!inputs.get(iterator).isEmpty()){
				inputs.get(iterator).put(frame);
				return null;
			}
			cancel();
			process1(frame);
		}
		
		process1(frame);
		return null;
	}

	private void process1(F frame) {
		List<Element> sliceLocs = new ArrayList<>();
		
		if (firstQueue.isEmpty()) fillQueue();
		for (Element element : firstQueue){
			if (element instanceof FrameElements){
				FrameElements fe = (FrameElements) element;
				if (fe.getNumber()==frame.getFrameNumber()){
					for ( Element l : fe.getList())
						sliceLocs.add(l);
					
					firstQueue.remove(element);
					Element newEl = locInput.get();
					if (newEl!=null) 
						firstQueue.add(newEl);		
				}
			}
		}
		
		if (sliceLocs.isEmpty()){
			secondQueue.put(frame);
			return;
		}	
		counter.addAndGet(sliceLocs.size());
		FrameElements res = fit( sliceLocs, frame.getPixels(), size, frame.getFrameNumber());
		for (Element el : res.getList())
			newOutput(el);
	}
	
	public abstract FrameElements fit(List<Element> sliceLocs, RandomAccessibleInterval<T> pixels, long windowSize, long frameNumber);
		

	@SuppressWarnings({ "unchecked" })
	@Override
	protected void afterRun() {
		Store finisherList = new FastStore(); 
		if(!firstQueue.isEmpty())
			for (Element q : firstQueue)
				finisherList.put(q);
		firstQueue.clear();
		while (!locInput.isEmpty()){
			Element q = locInput.get();
			if (q != null)
				finisherList.put(q);
		}
		Collection<Element> view = finisherList.view();

		List<Element> sliceLocs = new ArrayList<>();
		
		int secondRun = secondQueue.getLength();
		
		while (!secondQueue.isEmpty()){
			F frame = (F) secondQueue.get();
			for (Element element : view){
				if (element instanceof FrameElements){
					FrameElements fe = (FrameElements) element;
					if (fe.getNumber()==frame.getFrameNumber()){
						for ( Element l : fe.getList())
							sliceLocs.add(l);
					}
				}
			}
			counter.addAndGet(sliceLocs.size());
			if (!sliceLocs.isEmpty()){
				FrameElements res = fit(sliceLocs,frame.getPixels(), size, frame.getFrameNumber());
				for (Element el : res.getList())
					newOutput(el);
				sliceLocs.clear();
			}
		}
		FittedLocalization lastLoc = new FittedLocalization(0, -1, -1, 0, -1, -1) ;
		lastLoc.setLast(true);
		newOutput(lastLoc);
		System.out.println("Fitting of "+ counter +" elements done in " + (System.currentTimeMillis() - start) + "ms with elements in 2nd run:" + secondRun);
	}

	@Override
	public boolean check() {
		return inputs.size()==2 && outputs.size()>=1;
	}
	
	protected static Roi cropRoi(Rectangle imageRoi, Rectangle curRect) {
		double x1 = curRect.getMinX() < imageRoi.getMinX() ? imageRoi.getMinX() : curRect.getMinX();
		double y1 = curRect.getMinY() < imageRoi.getMinY() ? imageRoi.getMinY() : curRect.getMinY();
		double x2 = curRect.getMaxX() > imageRoi.getMaxX() ? imageRoi.getMaxX() : curRect.getMaxX();
		double y2 = curRect.getMaxY() > imageRoi.getMaxY() ? imageRoi.getMaxY() : curRect.getMaxY();
		return new Roi(x1,y1,x2-x1,y2-y1);
	}


}
