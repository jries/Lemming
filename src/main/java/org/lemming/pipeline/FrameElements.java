package org.lemming.pipeline;

import java.util.List;

import org.lemming.interfaces.Element;

public class FrameElements implements Element {
	
	private boolean isLast;
	private List<Element> list;
	private long number;

	public FrameElements(List<Element> list_, long num) {
		list = list_;
		number = num;
	}

	@Override
	public boolean isLast() {
		return isLast;
	}

	@Override
	public void setLast(boolean b) {
		isLast = b;
	}

	@Override
	public Element deepClone() {
		return new FrameElements(list, number);
	}

	public long getNumber(){
		return number;
	}
	
	public List<Element> getList(){
		return list;
	}
}
